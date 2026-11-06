"""FINAL Phase 2 run on eval-v1: the chosen pipeline, through the HTTP query service and Pinecone.

Run this ONCE, after every choice has been made on the dev set. It refuses to run again if its
output file exists (so nobody is tempted to tune against eval-v1); delete the file on purpose to redo.

Writes results/phase2_eval_v1.json
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.metrics import relative_improvement  # noqa: E402
from src.eval.score import load_context, read_jsonl, score_rankings  # noqa: E402
from src.services import common  # noqa: E402

TARGET = 0.6175


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-url", default="http://127.0.0.1:8004")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "phase2_eval_v1.json")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out.name} already exists: eval-v1 was already run for Phase 2. Not running again.")

    gold = list(read_jsonl(ROOT / "data" / "eval" / "gold_questions.jsonl"))
    ctx = load_context(common.DATA_DIR)
    baseline = json.loads((ROOT / "results" / "baseline.json").read_text())
    rankings = []
    with httpx.Client(base_url=args.query_url, timeout=120) as http:
        health = http.get("/health").json()
        assert health.get("pipeline") == "v2", f"query service is not running the v2 pipeline: {health}"
        for g in gold:
            r = http.post("/query", json={"question": g["question"], "top_k": 10})
            r.raise_for_status()
            rankings.append([(x["set_id"], x["start"], x["end"]) for x in r.json()["results"]])
    sc = score_rankings(gold, rankings, ctx)
    rel = sc["relaxed"]
    base = {m: baseline[m] for m in ("recall_at_1", "recall_at_5", "mrr_at_10")}
    out = {
        "what": "Phase 2 pipeline on frozen eval-v1 via HTTP query service + Pinecone (run once)",
        "pipeline_config": json.loads(common.PIPELINE_CONFIG.read_text()),
        "n_questions": len(gold),
        "relaxed_primary": rel,
        "baseline_eval_v1": base,
        "absolute_improvement": {m: round(rel[m] - base[m], 4) for m in base},
        "relative_improvement": {m: round(relative_improvement(rel[m], base[m]), 4) for m in base},
        "target_recall_at_5": TARGET,
        "meets_target": bool(rel["recall_at_5"] >= TARGET),
        "strict_crosscheck": {"baseline": baseline["strict_crosscheck"], "phase2": sc["strict"]},
        "per_section_recall_at_5": sc["per_section_recall_at_5"],
        "per_section_n": sc["per_section_n"],
        "health": health,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("relaxed_primary", "baseline_eval_v1", "absolute_improvement",
                                         "relative_improvement", "meets_target")}, indent=2))


if __name__ == "__main__":
    main()
