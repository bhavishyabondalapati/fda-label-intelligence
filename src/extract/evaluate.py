"""Score saved extraction runs against gold. Pure functions, no model calls.

records: one dict per label from scripts/run_extraction.py (has "set_id", "validation", "extraction").
labels:  {set_id: labels.jsonl record} (the gold lives here; the workflow never saw it).
"""
from src.extract.scoring import FIELDS, f1_from_counts, gold_fields, pred_fields, strict_correct
from src.extract.validate import route_decisions

CONF_SETTINGS = ["low", "medium", "high"]       # "low" = no confidence requirement (gates only)


def values_of(rec):
    """The model's raw values {field: value} (None if its reply was not valid JSON)."""
    ex = rec.get("extraction")
    if not isinstance(ex, dict):
        return {}
    return {f: (ex.get(f) or {}).get("value") if isinstance(ex.get(f), dict) else None for f in FIELDS}


def evaluate(records, labels, min_conf):
    rows = []                                   # one row per (label, field)
    per_label_ok = []
    for rec in records:
        lab = labels[rec["set_id"]]
        gold, pred = gold_fields(lab), pred_fields(values_of(rec))
        dec = route_decisions(rec["validation"], min_conf)
        all_ok = True
        from src.extract.scoring import field_correct, field_items
        for f in FIELDS:
            approved = dec[f]["approved"]
            correct = field_correct(f, pred[f], gold[f])
            rows.append({"set_id": rec["set_id"], "field": f, "approved": approved, "correct": correct,
                         "items": field_items(f, pred[f], gold[f])})
            all_ok = all_ok and approved
        per_label_ok.append(all_ok)

    def rate(sel):
        return sum(sel) / len(sel) if sel else 0.0

    n = len(rows)
    approved = [r for r in rows if r["approved"]]
    wrong = [r for r in rows if not r["correct"]]
    out = {
        "min_conf": min_conf,
        "n_labels": len(records),
        "n_field_values": n,
        "per_field_auto_approval_rate": rate([r["approved"] for r in rows]),
        "per_label_auto_approval_rate": rate(per_label_ok),
        "auto_approved_precision": rate([r["correct"] for r in approved]),
        "n_auto_approved": len(approved),
        "n_wrong_extractions": len(wrong),
        "wrong_sent_to_review": sum(1 for r in wrong if not r["approved"]),
        "queue_catch_rate": rate([not r["approved"] for r in wrong]),
        "queue_precision": rate([not r["correct"] for r in rows if not r["approved"]]),
        "accuracy_all_fields": rate([r["correct"] for r in rows]),
    }
    tp = sum(r["items"][0] for r in rows); fp = sum(r["items"][1] for r in rows); fn = sum(r["items"][2] for r in rows)
    out["field_f1_micro"] = f1_from_counts(tp, fp, fn)
    out["by_field"] = {}
    for f in FIELDS:
        fr = [r for r in rows if r["field"] == f]
        fa = [r for r in fr if r["approved"]]
        ftp, ffp, ffn = (sum(r["items"][i] for r in fr) for i in range(3))
        out["by_field"][f] = {
            "auto_approval_rate": rate([r["approved"] for r in fr]),
            "auto_approved_precision": rate([r["correct"] for r in fa]),
            "n_auto_approved": len(fa), "accuracy": rate([r["correct"] for r in fr]),
            "f1": f1_from_counts(ftp, ffp, ffn)["f1"]}
    strict = {}
    for f in ("generic_name", "active_ingredients"):
        strict[f] = rate([strict_correct(f, values_of(r).get(f), labels[r["set_id"]]) for r in records])
    out["strict_accuracy_no_salt_stripping"] = strict
    return out


def choose_setting(results, min_precision=0.96):
    """Pre-registered rule: highest per-field auto-approval rate with train precision >= min_precision.
    If none qualifies, the strictest setting. Returns (setting, note)."""
    ok = [r for r in results if r["auto_approved_precision"] >= min_precision and r["n_auto_approved"] > 0]
    if ok:
        best = max(ok, key=lambda r: (r["per_field_auto_approval_rate"], CONF_SETTINGS.index(r["min_conf"])))
        return best["min_conf"], f"highest approval rate with train precision >= {min_precision:.2f}"
    return CONF_SETTINGS[-1], f"no setting reached train precision {min_precision:.2f}; strictest used"
