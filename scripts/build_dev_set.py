"""Build the DEV question set -> data/eval/dev_questions.jsonl  (Phase 2 tuning set).

Why: eval-v1 is the final exam. Every Phase 2 tuning choice is made on THIS set instead, so the
number reported on eval-v1 is not fitted to those 200 questions.

Same templates, same hit rule, same builder as eval-v1, but:
  * questions come from `extraction_train` labels (never from retrieval_eval),
  * any label whose (drug name, route) also occurs among ALL retrieval_eval labels is excluded,
    so dev and eval-v1 share no drug at all (generics with the same name would leak otherwise).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.gold import build_gold, drug_key  # noqa: E402
from src.eval.score import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "dev_questions.jsonl")
    args = ap.parse_args()

    labels = list(read_jsonl(args.data_dir / "labels.jsonl"))
    eval_keys = {drug_key(lab) for lab in labels if lab["split"] == "retrieval_eval"}
    wanted = {lab["set_id"] for lab in labels if lab["split"] == "extraction_train" and drug_key(lab) not in eval_keys}
    sections = defaultdict(dict)
    for s in read_jsonl(args.data_dir / "sections.jsonl"):
        if s["set_id"] in wanted:
            sections[s["set_id"]][s["section"]] = s["text"]
    gold = build_gold(labels, sections, n_questions=args.n, seed=args.seed,
                      split="extraction_train", exclude_keys=eval_keys)
    if len(gold) < args.n:
        sys.exit(f"Only {len(gold)} dev questions could be built")
    with open(args.out, "w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    # safety checks: no overlap with eval-v1
    ev = list(read_jsonl(ROOT / "data" / "eval" / "gold_questions.jsonl"))
    by_id = {lab["set_id"]: lab for lab in labels}
    assert not {g["set_id"] for g in gold} & {e["set_id"] for e in ev}
    assert not {drug_key(by_id[g["set_id"]]) for g in gold} & eval_keys
    print(f"Wrote {len(gold)} dev questions to {args.out}")
    print("By section:", dict(Counter(g["section"] for g in gold)))
    print("Route in text:", sum(g["route_in_question"] for g in gold), "| distinct drugs:", len({g['drug_name'] for g in gold}))
    print("Overlap with eval-v1 questions (text):", len({g['question'] for g in gold} & {e['question'] for e in ev}))


if __name__ == "__main__":
    main()
