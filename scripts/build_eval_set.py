"""Build the gold retrieval questions -> data/eval/gold_questions.jsonl

Reads data/processed/{labels,sections}.jsonl (made by download_subset.py).
Questions come ONLY from labels in the `retrieval_eval` split, so the fine-tuning
data (extraction_train) can never overlap with the eval data.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.gold import build_gold  # noqa: E402


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "gold_questions.jsonl")
    args = ap.parse_args()

    labels = list(read_jsonl(args.data_dir / "labels.jsonl"))
    wanted = {lab["set_id"] for lab in labels if lab["split"] == "retrieval_eval"}
    sections = defaultdict(dict)
    for s in read_jsonl(args.data_dir / "sections.jsonl"):
        if s["set_id"] in wanted:
            sections[s["set_id"]][s["section"]] = s["text"]

    gold = build_gold(labels, sections, n_questions=args.n, seed=args.seed)
    if len(gold) < args.n:
        sys.exit(f"Only {len(gold)} questions could be built (wanted {args.n}). "
                 "Use a larger subset or a smaller --n.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    print(f"Wrote {len(gold)} questions to {args.out}")
    print("By section:", dict(Counter(g["section"] for g in gold)))
    print("Distinct drugs:", len({g["drug_name"] for g in gold}))


if __name__ == "__main__":
    main()
