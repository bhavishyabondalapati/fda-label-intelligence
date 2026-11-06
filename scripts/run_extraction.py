"""Run the LangGraph extraction workflow over a sample of labels and save every result (resumable).

  python scripts/run_extraction.py --split extraction_train --n 100 --out results/extract_train_raw.jsonl

Sample: labels of the split sorted by sha256("<salt>:<set_id>"), first N (fixed, no randomness to tune).
The run uses min_conf=low (gates only) so confidence thresholds can be applied afterwards for free.
The first --summaries labels that have flagged fields also run the summarize node (cost measurement).
Bedrock spend goes to results/bedrock_usage_phase3.jsonl with a hard cap (--budget).
"""
import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.extract.graph import build_graph  # noqa: E402
from src.extract.retrieve import load_sections  # noqa: E402
from src.generation.usage import BudgetExceeded, UsageLedger  # noqa: E402
from src.services import common  # noqa: E402  (loads .env)

LEDGER = ROOT / "results" / "bedrock_usage_phase3.jsonl"


def sample_ids(labels, split, n, salt):
    ids = [l["set_id"] for l in labels if l["split"] == split]
    return sorted(ids, key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())[:n]


def main():
    import os
    from langchain_aws import ChatBedrockConverse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="extraction_train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--salt", default="phase3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summaries", type=int, default=0, help="run the summarize node for the first K labels")
    ap.add_argument("--min-conf", default="low", help="confidence setting used for routing in this run")
    ap.add_argument("--budget", type=float, default=1.50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first K of the sample (smoke test)")
    args = ap.parse_args()

    labels = [json.loads(l) for l in open(common.DATA_DIR / "labels.jsonl")]
    ids = sample_ids(labels, args.split, args.n, f"{args.salt}-{args.split}")
    if args.limit:
        ids = ids[:args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["set_id"] for l in open(args.out) if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} labels in sample, {len(done)} already done, {len(todo)} to run")

    model = os.environ["BEDROCK_MODEL_ID"]
    ledger = UsageLedger(LEDGER, args.budget, float(os.environ["BEDROCK_PRICE_IN_PER_M"]),
                         float(os.environ["BEDROCK_PRICE_OUT_PER_M"]), model)
    llm = ChatBedrockConverse(model=model, region_name=os.getenv("AWS_REGION", "us-east-2"),
                              credentials_profile_name=os.getenv("AWS_PROFILE", "sky"), temperature=0, max_tokens=900)
    sections = load_sections(common.DATA_DIR / "sections.jsonl", todo)
    graph = build_graph(llm, ledger, sections)

    lock, summary_slots = threading.Lock(), {"left": args.summaries}

    def work(sid):
        with lock:
            want_summary = summary_slots["left"] > 0
        out = graph.invoke({"set_id": sid, "min_conf": args.min_conf, "do_summary": want_summary})
        if out.get("summary"):
            with lock:
                summary_slots["left"] -= 1
        rec = {"set_id": sid, "n_source_chars": len(out["source_text"]), "raw_reply": out["raw_reply"],
               "extraction": out["extraction"], "validation": out["validation"], "decisions": out["decisions"],
               "summary": out.get("summary"), "usage": out["usage"]}
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return sid

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            for k, _ in enumerate(ex.map(work, todo), 1):
                if k % 10 == 0:
                    print(f"  {k}/{len(todo)} done, spent ${ledger.total():.4f}", flush=True)
    except BudgetExceeded as e:
        print("STOPPED:", e)
    print(f"phase 3 ledger total: ${ledger.total():.4f}")


if __name__ == "__main__":
    main()
