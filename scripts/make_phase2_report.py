"""Assemble results/phase2.json (and a markdown table for the README) from the saved result files."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
abl = json.loads((R / "ablation_dev.json").read_text())
rows = {r["config"]: r for r in abl["results"]}
base = rows["baseline"]
rng = np.random.default_rng(0)


def ci_diff(cfg):            # paired bootstrap of recall@5 difference vs baseline on dev
    a, b = np.array(rows[cfg]["hit5"], float), np.array(base["hit5"], float)
    d = [(a[i] - b[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def line(name, cfg):
    r = rows[cfg]
    return {"step": name, "config": cfg, "recall_at_1": r["recall_at_1"], "recall_at_5": r["recall_at_5"],
            "mrr_at_10": r["mrr_at_10"],
            "recall_at_5_rel_to_dev_baseline": round((r["recall_at_5"] - base["recall_at_5"]) / base["recall_at_5"], 3)}


one_at_a_time = [line("baseline (naive chunks, dense)", "baseline"), line("+ headers only", "H"),
                 line("+ section-aware chunking only", "S"), line("+ drug filter only", "F"),
                 line("+ hybrid BM25+RRF only", "B"), line("+ reranker only", "R")]
path = [line("baseline", "baseline"), line("+ headers", "H"), line("+ headers + reranker", "HR"),
        line("+ headers + reranker + drug filter  <- CHOSEN", "HFR")]
no_filter = max((r for k, r in rows.items() if "F" not in k and k != "baseline"), key=lambda r: (r["recall_at_5"], r["mrr_at_10"]))
stage_ms = abl["stage_ms_per_query"]
out = {
    "ablation_set": "dev_questions.jsonl (400 questions, drugs disjoint from eval-v1)",
    "selection_rule": "highest dev recall@5; configs within 0.005 tie; then fewest switches; then MRR",
    "one_at_a_time_dev": one_at_a_time,
    "greedy_path_dev": path,
    "all_32_configs_dev": [{k: v for k, v in r.items() if k not in ("hit5", "switches")} for r in abl["results"]],
    "best_without_drug_filter_dev": {k: no_filter[k] for k in ("config", "recall_at_1", "recall_at_5", "mrr_at_10")},
    "chosen_vs_dev_baseline_recall_at_5_diff_95ci": ci_diff("HFR"),
    "stage_latency_ms": {"bm25_per_query_local": stage_ms["bm25"], "rerank_top50_per_query_local": stage_ms["rerank_top50"],
                         "drug_detect_per_query": stage_ms["drug_detect"]},
    "chosen_config": json.loads((R.parent / "data/processed/pipeline_v2.json").read_text()),
    "messy_dev": json.loads((R / "messy_dev.json").read_text())["configs"],
    "eval_v1": json.loads((R / "phase2_eval_v1.json").read_text()),
    "search_latency": {k: v for k, v in json.loads((R / "latency_phase2_search.json").read_text()).items()
                       if k in ("p95_ms", "vectors_in_index", "warm", "n_timed_queries", "namespace")},
}
if (R / "generation_phase2.json").exists():
    g = json.loads((R / "generation_phase2.json").read_text())
    out["generation"] = {k: v for k, v in g.items() if k != "examples"}
(R / "phase2.json").write_text(json.dumps(out, indent=1))
t = ["| Step (dev, 400 q) | recall@1 | recall@5 | MRR@10 |", "|---|---|---|---|"]
for r in one_at_a_time + path[1:]:
    t.append(f"| {r['step']} | {r['recall_at_1']:.3f} | {r['recall_at_5']:.3f} | {r['mrr_at_10']:.3f} |")
print("\n".join(t))
print("best without filter:", out["best_without_drug_filter_dev"], "| CI of chosen-baseline R@5:", out["chosen_vs_dev_baseline_recall_at_5_diff_95ci"])
print(out["stage_latency_ms"])
