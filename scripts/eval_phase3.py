"""Score the single extraction_test run (free, no model calls) and write results/phase3.json.

Chosen confidence setting = the one picked on TRAIN by the pre-registered rule (results/phase3_train.json).
Also folds in results/phase3_audit.json (a manual audit of approved-but-disagreeing values) if it exists;
the audit never changes the official numbers.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.extract.evaluate import CONF_SETTINGS, evaluate, values_of  # noqa: E402
from src.extract.scoring import FIELDS, gold_fields, pred_fields  # noqa: E402
from src.extract.validate import route_decisions  # noqa: E402


def disagreements(recs, labels, min_conf):
    """Approved values that differ from openFDA gold, with the context needed to judge them."""
    rows = []
    for r in recs:
        lab = labels[r["set_id"]]
        g, p = gold_fields(lab), pred_fields(values_of(r))
        d = route_decisions(r["validation"], min_conf)
        for f in FIELDS:
            if d[f]["approved"] and p[f] != g[f]:
                rows.append({"set_id": r["set_id"], "field": f, "predicted": values_of(r).get(f),
                             "quotes": r["validation"][f]["quotes"][:2],
                             "openfda": {"generic_name": lab["generic_name"], "substance_name": lab["substance_name"],
                                         "route": lab["route"], "pharm_class_epc": lab["pharm_class_epc"],
                                         "has_boxed_warning_section": "boxed_warning" in lab["sections"]}})
    return rows


def main():
    labels = {}
    for line in open(ROOT / "data" / "processed" / "labels.jsonl"):
        d = json.loads(line)
        labels[d["set_id"]] = d
    recs = [json.loads(l) for l in open(ROOT / "results" / "extract_test_raw.jsonl") if l.strip()]
    assert len(recs) == 150 and all(labels[r["set_id"]]["split"] == "extraction_test" for r in recs)
    train = json.loads((ROOT / "results" / "phase3_train.json").read_text())
    chosen = train["chosen_min_conf"]
    settings = {m: evaluate(recs, labels, m) for m in CONF_SETTINGS}
    main_res = settings[chosen]

    usage = [u for r in recs for u in r["usage"]]
    ext = [u for u in usage if u["tag"] == "extract"]
    summ = [u for u in usage if u["tag"] == "summarize"]
    cost = {
        "labels": len(recs), "extract_cost_usd": round(sum(u["cost_usd"] for u in ext), 4),
        "summarize_calls": len(summ), "summarize_cost_usd": round(sum(u["cost_usd"] for u in summ), 4),
        "test_run_cost_usd": round(sum(u["cost_usd"] for u in usage), 4),
        "per_label_usd": round(sum(u["cost_usd"] for u in usage) / len(recs), 5),
        "phase3_ledger_total_usd": round(sum(json.loads(l)["cost_usd"] for l in open(ROOT / "results" / "bedrock_usage_phase3.jsonl")), 4),
        "budget_cap_usd": 1.50,
        "extract_latency_ms": {"p50": round(statistics.median(u["ms"] for u in ext)),
                               "max": round(max(u["ms"] for u in ext))},
        "mean_input_tokens": round(statistics.mean(u["input_tokens"] for u in ext)),
        "mean_output_tokens": round(statistics.mean(u["output_tokens"] for u in ext)),
    }
    dis = disagreements(recs, labels, chosen)
    # The headline catch rate is inflated by pharm_class (always reviewed). Also report it without that field.
    wrong_other = caught_other = 0
    for r in recs:
        g, p = gold_fields(labels[r["set_id"]]), pred_fields(values_of(r))
        d = route_decisions(r["validation"], chosen)
        for f in FIELDS:
            if f != "pharm_class" and p[f] != g[f]:
                wrong_other += 1
                caught_other += not d[f]["approved"]
    catch_without_review_only = {"wrong_values": wrong_other, "sent_to_review": caught_other,
                                 "rate": round(caught_other / wrong_other, 4) if wrong_other else None}
    audit_path = ROOT / "results" / "phase3_audit.json"
    out = {
        "what": "Phase 3: LangGraph extraction + validation + review queue, Claude Haiku 4.5, 150 extraction_test labels, run once",
        "baseline": "100% manual review = 0% auto-approved",
        "chosen_confidence_setting": chosen, "choice_made_on": "extraction_train (pre-registered rule)",
        "test": main_res, "test_other_confidence_settings": {m: v for m, v in settings.items() if m != chosen},
        "targets": {"auto_approval_per_field_ge": 0.40, "auto_approved_precision_ge": 0.95},
        "target_check": {"auto_approval_met": main_res["per_field_auto_approval_rate"] >= 0.40,
                         "precision_met": main_res["auto_approved_precision"] >= 0.95,
                         "precision_ci95_lower_ge_0.95": main_res["auto_approved_precision_ci95"][0] >= 0.95},
        "invalid_json_replies": sum(not isinstance(r["extraction"], dict) for r in recs),
        "cost": cost,
        "train": {k: train[k] for k in ("n_labels", "chosen_min_conf", "choice_rule", "cost")} | {"chosen_setting_result": next(s for s in train["settings"] if s["min_conf"] == chosen)},
        "queue_catch_excluding_pharm_class": catch_without_review_only,
        "n_approved_but_disagreeing_with_openfda": len(dis),
        "audit": json.loads(audit_path.read_text()) if audit_path.exists() else None,
        "caveats": [
            "Simulated review workload on public labels; 'correct' means equal to the openFDA structured data under the pre-registered matching rules.",
            "Gold values come from openFDA and are sometimes incomplete or wrong (see audit); official numbers are scored against openFDA only.",
            "has_boxed_warning is close to a lookup: a boxed-warning section, when it exists, is part of the text the model reads.",
            "pharm_class is always reviewed by policy (amendment 2026-11-06); it is scored for accuracy and F1 but never auto-approved.",
            "Model self-rated confidence carried almost no information (99% 'high' on most fields); the verbatim-quote check is the real gate.",
            "Rules were amended once, after the train run and before any test output was seen (PLAN.md).",
        ],
    }
    (ROOT / "results" / "phase3.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "phase3_disagreements.json").write_text(json.dumps(dis, indent=2))
    t = main_res
    print(f"setting {chosen}: per-field approved {t['per_field_auto_approval_rate']:.3f}, per-label(5 fields) {t['per_label_auto_approval_rate']:.3f}")
    print(f"precision {t['auto_approved_precision']:.3f} CI {t['auto_approved_precision_ci95']} (n={t['n_auto_approved']}); wrong values sent to review {t['wrong_sent_to_review']}/{t['n_wrong_extractions']} = {t['queue_catch_rate']:.3f}")
    print(f"F1 micro {t['field_f1_micro']['f1']:.3f}  accuracy all fields {t['accuracy_all_fields']:.3f}")
    for f, v in t["by_field"].items():
        print(f"  {f:20s} approve {v['auto_approval_rate']:.2f} prec {v['auto_approved_precision']:.2f} CI {v['auto_approved_precision_ci95']} (n={v['n_auto_approved']}) acc {v['accuracy']:.2f} f1 {v['f1']:.2f}")
    for m, v in settings.items():
        print(f"  [{m}] field-approval {v['per_field_auto_approval_rate']:.3f} precision {v['auto_approved_precision']:.3f}")
    print("strict (no salt/form stripping):", t["strict_accuracy_no_salt_stripping"])
    print("cost", cost, "invalid json", out["invalid_json_replies"], "disagreements", len(dis))


if __name__ == "__main__":
    main()
