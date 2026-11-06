"""Apply the pre-registered threshold rule to the saved TRAIN run (no model calls, free).

  python scripts/tune_extraction.py   ->  results/phase3_train.json
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.extract.evaluate import CONF_SETTINGS, choose_setting, evaluate  # noqa: E402
from src.extract.scoring import FIELDS  # noqa: E402


def main():
    labels = {}
    for line in open(ROOT / "data" / "processed" / "labels.jsonl"):
        d = json.loads(line)
        labels[d["set_id"]] = d
    recs = [json.loads(l) for l in open(ROOT / "results" / "extract_train_raw.jsonl") if l.strip()]
    assert all(labels[r["set_id"]]["split"] == "extraction_train" for r in recs), "train run only"
    results = [evaluate(recs, labels, m) for m in CONF_SETTINGS]
    chosen, note = choose_setting(results)

    usage = [u for r in recs for u in r["usage"]]
    ext = [u for u in usage if u["tag"] == "extract"]
    summ = [u for u in usage if u["tag"] == "summarize"]
    conf_dist = {f: {c: sum(r["validation"][f]["confidence"] == c for r in recs) for c in CONF_SETTINGS} for f in FIELDS}
    out = {
        "split": "extraction_train", "n_labels": len(recs),
        "chosen_min_conf": chosen, "choice_rule": note,
        "settings": results,
        "confidence_distribution": conf_dist,
        "invalid_json_replies": sum(not isinstance(r["extraction"], dict) for r in recs),
        "cost": {
            "extract_calls": len(ext), "extract_cost_usd": round(sum(u["cost_usd"] for u in ext), 4),
            "mean_extract_input_tokens": round(statistics.mean(u["input_tokens"] for u in ext)),
            "mean_extract_output_tokens": round(statistics.mean(u["output_tokens"] for u in ext)),
            "summarize_calls": len(summ), "summarize_cost_usd": round(sum(u["cost_usd"] for u in summ), 4),
            "mean_summarize_cost_usd": round(statistics.mean(u["cost_usd"] for u in summ), 5) if summ else None,
            "extract_latency_ms_p50": round(statistics.median(u["ms"] for u in ext)),
        },
    }
    (ROOT / "results" / "phase3_train.json").write_text(json.dumps(out, indent=2))
    for r in results:
        print(f"{r['min_conf']:6s} field-approval {r['per_field_auto_approval_rate']:.3f}  label-approval {r['per_label_auto_approval_rate']:.3f}  "
              f"precision {r['auto_approved_precision']:.3f} (n={r['n_auto_approved']})  catch {r['queue_catch_rate']:.3f}  F1 {r['field_f1_micro']['f1']:.3f}")
    print("chosen:", chosen, "-", note)
    for f, v in results[0]["by_field"].items():
        print(f"  {f:20s} approve {v['auto_approval_rate']:.2f}  precision {v['auto_approved_precision']:.2f}  acc {v['accuracy']:.2f}  f1 {v['f1']:.2f}")
    print("conf dist", json.dumps(conf_dist)); print("cost", out["cost"], "invalid json", out["invalid_json_replies"])
    print("strict", results[0]["strict_accuracy_no_salt_stripping"])


if __name__ == "__main__":
    main()
