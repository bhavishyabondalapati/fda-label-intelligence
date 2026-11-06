"""Generation check through the /answer endpoint (needs: make up with PIPELINE=v2, make up-answer, aws login).

  * 15 ANSWERABLE questions (from the dev set): does the answer cite chunks that were really retrieved?
  * 10 UNANSWERABLE questions (drugs that are not in the corpus, or off-topic): does it say
    "Not enough information"?
  * End-to-end latency (search + generation) for all 25 calls, next to the search-only part.
Cost is limited by the ledger cap (BEDROCK_BUDGET_USD). Writes results/generation_phase2.json
"""
import json
import random
import statistics
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402

UNANSWERABLE = [
    "What is Ozempic indicated for?", "What are the side effects of Zyxomadrine?",
    "What is the recommended dosage of Wegovy?", "How does Keytruda work in the body?",
    "What is the capital of France?", "Who won the 2018 World Cup?",
    "What are the contraindications for Qwertyvex?", "How should Dupixent be stored?",
    "Explain how to change a car tire.", "What is the boxed warning for Blorfinib?",
]


def main():
    dev = list(read_jsonl(ROOT / "data" / "eval" / "dev_questions.jsonl"))
    random.Random(1).shuffle(dev)
    answerable = [g["question"] for g in dev[:15]]
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8005", timeout=120) as http:
        for kind, qs in (("answerable", answerable), ("unanswerable", UNANSWERABLE)):
            for q in qs:
                r = http.post("/answer", json={"question": q, "top_k": 5})
                r.raise_for_status()
                rows.append({"kind": kind, **r.json()})
                print(f"[{kind}] {q[:60]!r} -> answerable={rows[-1]['answerable']} cites={len(rows[-1]['citations'])}", flush=True)
    ans = [r for r in rows if r["kind"] == "answerable"]
    un = [r for r in rows if r["kind"] == "unanswerable"]
    e2e = [r["end_to_end_ms"] for r in rows]
    search = [r["search_ms"] for r in rows]
    out = {
        "answerable": {"n": len(ans), "answered": sum(r["answerable"] for r in ans),
                       "with_valid_citation": sum(bool(r["citations"]) for r in ans if r["answerable"]),
                       "with_invalid_citation": sum(bool(r["invalid_citations"]) for r in rows)},
        "unanswerable": {"n": len(un), "correctly_refused": sum(not r["answerable"] for r in un)},
        "latency_ms": {"end_to_end_p50": round(statistics.median(e2e), 0), "end_to_end_max": round(max(e2e), 0),
                       "search_part_p50": round(statistics.median(search), 0)},
        "tokens": {"input_total": sum(r["usage"]["input_tokens"] for r in rows),
                   "output_total": sum(r["usage"]["output_tokens"] for r in rows),
                   "est_cost_usd": round(sum(r["usage"]["cost_usd"] for r in rows), 5)},
        "examples": rows,
    }
    (ROOT / "results" / "generation_phase2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
