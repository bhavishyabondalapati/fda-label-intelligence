"""Build the Phase 4 training files (PLAN.md "M4 pre-registration"). Free, local, reads extraction_train ONLY.

Writes data/finetune/train.jsonl (1,200), val.jsonl (100) and results/phase4_data.json (counts and quote coverage).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402
from src.extract.retrieve import load_sections, select_text  # noqa: E402
from src.extract.scoring import name_key  # noqa: E402
from src.finetune.data import build_example  # noqa: E402

N_TRAIN, N_VAL, SALT = 1200, 100, "phase4-train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "finetune")
    args = ap.parse_args()
    labels = [l for l in read_jsonl(ROOT / "data" / "processed" / "labels.jsonl") if l["split"] == "extraction_train"]
    assert all(l["split"] == "extraction_train" for l in labels)
    order = sorted(labels, key=lambda l: hashlib.sha256(f"{SALT}:{l['set_id']}".encode()).hexdigest())
    picked = order[:N_TRAIN + N_VAL]
    secs = load_sections(ROOT / "data" / "processed" / "sections.jsonl", [l["set_id"] for l in picked])

    rows = []
    for l in picked:
        s = secs.get(l["set_id"], {})
        text, _ = select_text(s)
        ex = build_example(l, s, text)
        ex["set_id"] = l["set_id"]
        ex["generic_key"] = sorted(name_key(l["generic_name"]))
        rows.append(ex)
    train, val = rows[:N_TRAIN], rows[N_TRAIN:]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(args.out / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def share(key, part=train):
        return round(sum(r["stats"][key] for r in part) / len(part), 4)

    summary = {"n_train": len(train), "n_val": len(val), "salt": SALT,
               "quote_coverage_train": {k: share(k) for k in ("quote_generic", "quote_ingredients", "quote_route", "quote_boxed")},
               "pharm_class_unmasked_share_train": share("pharm_unmasked"),
               "route_other_share_train": share("route_other"),
               "train_generic_keys": len({tuple(r["generic_key"]) for r in train})}
    (ROOT / "results" / "phase4_data.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
