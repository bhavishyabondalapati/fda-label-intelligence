"""Score Qwen outputs on extraction_test (PLAN.md, M4). Free, local. Writes results/phase4.json.

Always scores the base model; also scores the adapter and the comparison once results/phase4_adapter_raw.jsonl exists.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402
from src.extract.scoring import FIELDS, f1_from_counts, name_key, score_label  # noqa: E402
from src.finetune.metrics import FIELDS5, bootstrap, counts, micro_f1, parse_reply, values_from  # noqa: E402

R = ROOT / "results"


def load(path, labels):
    rows = {json.loads(l)["set_id"]: json.loads(l) for l in open(path)}
    out = []
    for sid, lab in labels.items():
        obj = parse_reply(rows[sid]["reply"])
        out.append({"set_id": sid, "label": lab, "values": values_from(obj), "parsed": obj is not None,
                    "seconds": rows[sid]["seconds"]})
    return out


def f1s(recs, fields):
    return round(micro_f1([counts(r["values"], r["label"], fields) for r in recs]), 4) if recs else None


def per_field(recs):
    res = {}
    for f in FIELDS:
        tp = fp = fn = acc = 0
        for r in recs:
            s = score_label(r["values"], r["label"])[f]
            tp, fp, fn, acc = tp + s["items"][0], fp + s["items"][1], fn + s["items"][2], acc + s["correct"]
        res[f] = {"f1": round(f1_from_counts(tp, fp, fn)["f1"], 4), "accuracy": round(acc / len(recs), 4)}
    return res


def summarize(recs, seen_keys):
    unseen = [r for r in recs if name_key(r["label"]["generic_name"]) not in seen_keys["train_used"]]
    unseen_all = [r for r in recs if name_key(r["label"]["generic_name"]) not in seen_keys["train_all"]]
    parsed = [r for r in recs if r["parsed"]]
    return {"n": len(recs), "parse_rate": round(len(parsed) / len(recs), 4),
            "f1_6_fields": f1s(recs, FIELDS), "f1_5_fields": f1s(recs, FIELDS5),
            "parsed_only": {"n": len(parsed), "f1_6_fields": f1s(parsed, FIELDS), "f1_5_fields": f1s(parsed, FIELDS5)},
            "unseen_vs_1200_used": {"n": len(unseen), "f1_6_fields": f1s(unseen, FIELDS), "f1_5_fields": f1s(unseen, FIELDS5)},
            "unseen_vs_all_4058_train": {"n": len(unseen_all), "f1_6_fields": f1s(unseen_all, FIELDS),
                                         "f1_5_fields": f1s(unseen_all, FIELDS5)},
            "per_field": per_field(recs), "mean_seconds_per_label": round(sum(r["seconds"] for r in recs) / len(recs), 1)}


def compare(base, ad, seen_keys):
    out = {}
    def block(name, ra, rb, fields):
        fa, fb = micro_f1([counts(r["values"], r["label"], fields) for r in ra]), micro_f1([counts(r["values"], r["label"], fields) for r in rb])
        d = {"n": len(ra), "base": round(fa, 4), "adapter": round(fb, 4), "abs": round(fb - fa, 4),
             "rel": round((fb - fa) / fa, 4) if fa else None}
        d.update(bootstrap([counts(r["values"], r["label"], fields) for r in ra], [counts(r["values"], r["label"], fields) for r in rb]))
        out[name] = d
    block("all_150_6_fields (headline)", base, ad, FIELDS)
    block("all_150_5_fields", base, ad, FIELDS5)
    both = [(a, b) for a, b in zip(base, ad) if a["parsed"] and b["parsed"]]
    if both:
        block("both_parsed_6_fields", [a for a, _ in both], [b for _, b in both], FIELDS)
        block("both_parsed_5_fields", [a for a, _ in both], [b for _, b in both], FIELDS5)
    un = [(a, b) for a, b in zip(base, ad) if name_key(a["label"]["generic_name"]) not in seen_keys["train_used"]]
    if un:
        block("unseen_drugs_6_fields", [a for a, _ in un], [b for _, b in un], FIELDS)
        block("unseen_drugs_5_fields", [a for a, _ in un], [b for _, b in un], FIELDS5)
    return out


def main():
    labels_all = list(read_jsonl(ROOT / "data" / "processed" / "labels.jsonl"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_qwen_extraction import test_ids
    ids = test_ids(labels_all)
    labels = {l["set_id"]: l for l in labels_all if l["set_id"] in set(ids)}
    labels = {i: labels[i] for i in ids}
    train_all = {name_key(l["generic_name"]) for l in labels_all if l["split"] == "extraction_train"}
    used = {frozenset(r["generic_key"]) for r in read_jsonl(ROOT / "data" / "finetune" / "train.jsonl")}
    seen = {"train_used": used, "train_all": train_all}
    out = {"metric": "M4: micro field F1 on 150 extraction_test labels (PLAN.md)", "n_labels": len(labels),
           "haiku_reference_f1_6_fields": 0.787}
    base = load(R / "phase4_base_raw.jsonl", labels)
    out["base"] = summarize(base, seen)
    ad_path = R / "phase4_adapter_raw.jsonl"
    if ad_path.exists():
        ad = load(ad_path, labels)
        out["adapter"] = summarize(ad, seen)
        out["comparison"] = compare(base, ad, seen)
    (R / "phase4.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
