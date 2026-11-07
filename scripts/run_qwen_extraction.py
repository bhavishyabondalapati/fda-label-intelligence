"""Run Qwen2.5-1.5B-Instruct (base, or base + LoRA adapter) on the 150 extraction_test labels. Local, mps, free.

Same prompt as Phase 3, greedy decoding, up to 512 new tokens, bf16. Resumable: replies are appended to the output file.
Base and adapter must be run with the SAME settings (PLAN.md, M4). This is the ONE test run per model.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402
from src.extract.graph import EXTRACT_SYSTEM  # noqa: E402
from src.extract.retrieve import load_sections, select_text  # noqa: E402
from src.finetune.data import user_message  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def test_ids(labels, n=150):
    ids = [l["set_id"] for l in labels if l["split"] == "extraction_test"]
    return sorted(ids, key=lambda i: hashlib.sha256(f"phase3-extraction_test:{i}".encode()).hexdigest())[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter folder; omit for the base model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    labels = list(read_jsonl(ROOT / "data" / "processed" / "labels.jsonl"))
    ids = test_ids(labels, a.n)
    assert all(l["split"] == "extraction_test" for l in labels if l["set_id"] in set(ids))
    done = {json.loads(l)["set_id"] for l in open(a.out)} if a.out.exists() else set()
    secs = load_sections(ROOT / "data" / "processed" / "sections.jsonl", ids)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload()
    model.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        for k, sid in enumerate(ids, 1):
            if sid in done:
                continue
            text, _ = select_text(secs.get(sid, {}))
            prompt = tok.apply_chat_template([{"role": "system", "content": EXTRACT_SYSTEM},
                                              {"role": "user", "content": user_message(text)}],
                                             tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("mps")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None)
            reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"set_id": sid, "reply": reply, "prompt_tokens": int(enc["input_ids"].shape[1]),
                                "new_tokens": int(out.shape[1] - enc["input_ids"].shape[1]),
                                "seconds": round(time.time() - t0, 1), "adapter": str(a.adapter) if a.adapter else None}) + "\n")
            f.flush()
            print(f"{k}/{len(ids)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
