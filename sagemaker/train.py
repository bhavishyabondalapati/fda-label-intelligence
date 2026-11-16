"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct on the Phase 4 extraction data (PLAN.md "M4 pre-registration").

Runs on a free Kaggle T4 (fp16, 4-bit; see kaggle/), and locally for a smoke test (`--quant none --device mps`).
(It also reads SageMaker's SM_CHANNEL_TRAIN / SM_MODEL_DIR variables, left over from the unused SageMaker plan.) It is a plain PyTorch loop (no trainer library) so every step is visible.

Key ideas:
  * 4-bit base weights (nf4) + small trainable LoRA matrices: the base model is frozen, only ~1% is trained.
  * Loss only on the ANSWER tokens (the JSON), never on the long label text.
  * Tokens listed in `mask_spans` (pharm_class when its gold is unreliable) are left out of the loss.
  * Only the last k positions go through the output layer (`logits_to_keep`), which saves GBs of memory.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(tok, ex, max_len):
    """-> (input_ids, weights, first) or None if too long. weights[i] = 1 for supervised tokens, else 0."""
    prompt = tok.apply_chat_template([{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}],
                                     tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(ex["completion"] + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
    ids = p_ids + c["input_ids"]
    if len(ids) > max_len:
        return None
    w = [0.0] * len(p_ids)
    for tid, (a, b) in zip(c["input_ids"], c["offset_mapping"]):
        masked = any(a < e and b > s for s, e in ex["mask_spans"])
        w.append(0.0 if masked else 1.0)
    return torch.tensor(ids), torch.tensor(w), len(p_ids)


def example_loss(model, ids, w, first, device):
    """Mean cross-entropy over the supervised answer tokens of one example."""
    L = ids.shape[0]
    k = L - first + 1                                     # positions that predict tokens first..L-1
    out = model(input_ids=ids[None].to(device), logits_to_keep=k)
    logits = out.logits[0, :-1].float()                   # (k-1, V); the last position predicts nothing
    tgt = ids[first:].to(device)
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
    wt = w[first:].to(device)
    return (ce * wt).sum() / wt.sum().clamp(min=1.0)


def versions():
    """Library and GPU versions, saved with the results so the run can be reproduced."""
    import importlib.metadata as md
    v = {"torch": torch.__version__}
    for pkg in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
    return v


def main():
    env = os.environ
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=env.get("SM_CHANNEL_TRAIN", "data/finetune"))
    ap.add_argument("--out", default=env.get("SM_MODEL_DIR", "results/adapter"))
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-examples", type=int, default=0, help="smoke test only")
    ap.add_argument("--time-budget", type=float, default=6000, help="seconds of training before saving and stopping")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    print("versions:", json.dumps(versions()), flush=True)
    if a.quant == "4bit" and not torch.cuda.is_available():
        raise SystemExit("4-bit training needs a CUDA GPU: in Kaggle, set Accelerator to GPU T4 and restart the notebook")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"dtype": dt}
    if a.quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.quant != "4bit":
        model.to(a.device)
    model.config.use_cache = False
    if a.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    data_dir = Path(a.data)
    enc = [encode(tok, ex, a.max_len) for ex in read_jsonl(data_dir / "train.jsonl")]
    dropped = sum(e is None for e in enc)
    train = [e for e in enc if e]
    val = [e for e in (encode(tok, ex, a.max_len) for ex in read_jsonl(data_dir / "val.jsonl")) if e]
    random.shuffle(train)
    if a.max_examples:
        train, val = train[:a.max_examples], val[:2]
    n_tok = sum(len(e[0]) for e in train)
    print(f"train {len(train)} examples ({dropped} dropped as too long), {n_tok} tokens, val {len(val)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train) * a.epochs / a.accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(0.03 * total_steps)), total_steps)
    use_amp = a.device == "cuda" and a.dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    device = a.device

    model.train()
    t0, seen_tok, step, bad, losses, stopped_early = time.time(), 0, 0, 0, [], False
    order = [e for _ in range(a.epochs) for e in train]
    for i, (ids, w, first) in enumerate(order, 1):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = example_loss(model, ids, w, first, device)
        if not torch.isfinite(loss):
            bad += 1
            print(f"non-finite loss at example {i}", flush=True)
            if bad >= 3:
                raise SystemExit("3 non-finite losses in a row/total: stopping to save money (fp16 overflow?)")
            continue
        (scaler.scale(loss / a.accum) if scaler else loss / a.accum).backward()
        losses.append(loss.item()); seen_tok += len(ids)
        if i % a.accum == 0 or i == len(order):
            if scaler:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            opt.zero_grad(set_to_none=True); sched.step(); step += 1
            el = time.time() - t0
            if step % 10 == 0 or step == 1 or step == total_steps:
                rate = seen_tok / el
                print(f"step {step}/{total_steps} loss {sum(losses[-a.accum*10:])/len(losses[-a.accum*10:]):.4f} "
                      f"{rate:.0f} tok/s elapsed {el:.0f}s projected total {n_tok*a.epochs/rate:.0f}s", flush=True)
            if el > a.time_budget:
                stopped_early = True
                print("time budget reached: saving the adapter now", flush=True)
                break

    model.eval(); vl = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for ids, w, first in val:
            vl.append(example_loss(model, ids, w, first, device).item())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    metrics = {"model": a.model, "steps_done": step, "steps_planned": total_steps, "stopped_early": stopped_early,
               "train_examples": len(train), "dropped_too_long": dropped, "train_tokens": n_tok,
               "seconds": round(time.time() - t0, 1), "tokens_per_second": round(seen_tok / (time.time() - t0), 1),
               "first_10_loss": sum(losses[:10]) / max(1, len(losses[:10])), "last_10_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
               "val_loss": sum(vl) / max(1, len(vl)), "quant": a.quant, "dtype": a.dtype, "versions": versions()}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
