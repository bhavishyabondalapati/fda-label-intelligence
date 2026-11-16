# FDA Label Intelligence

A GenAI document-processing platform over public FDA drug labels (openFDA, public domain). It chunks, embeds, searches, and extracts structured fields from ~5K prescription drug labels. Every performance number in this repo is produced by a script in this repo, as a relative improvement over a stated baseline.

> **Status: Phases 0 to 4 complete (Phase 4: QLoRA adapter trained on a free Kaggle T4, evaluated once).** Eval set and hit rule are frozen as git tag `eval-v1`.
>
> Phase 0 was data, eval set and baseline. See [PLAN.md](PLAN.md) for the full roadmap and exact metric definitions. This README is a stub and is completed in Phase 5.

## Quick start (Phase 0)

```bash
conda create -n fda-label-intelligence python=3.11 -y
conda activate fda-label-intelligence
pip install -r requirements.txt

python scripts/download_subset.py      # ~1.8 GB download, builds data/processed/
python scripts/build_eval_set.py       # writes data/eval/gold_questions.jsonl
python scripts/run_baseline.py         # writes results/baseline.json
pytest                                 # tests
```

## Phase 1: run the services

```bash
make up PYTHON=$(which python)      # starts ingest :8001, chunk :8002, embed :8003, query :8004
python scripts/build_index.py       # chunk -> embed -> upsert to Pinecone (needs PINECONE_API_KEY in .env; ~25 min)
python scripts/eval_service.py      # parity: eval-v1 through the service vs the local exact baseline
make restart PYTHON=$(which python) && python scripts/bench_latency.py   # 500-query latency benchmark
make down
```

Results: `results/baseline.json`, `results/parity_phase1.json`, `results/latency_phase1.json`.

## Phase 2: better retrieval and cited answers

**Result on frozen eval-v1 (run once, through the HTTP service + Pinecone):** recall@5 **0.825** (baseline 0.475; +0.350 absolute, +73.7% relative), recall@1 0.720 (baseline 0.230), MRR@10 0.766 (baseline 0.331). Target was >= 0.6175.

All choices were made on a separate **dev set** (400 questions, drugs disjoint from eval-v1). Ablation on dev (each switch alone, then combined; all 32 combinations are in `results/phase2.json`):

| Step (dev, 400 q) | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| baseline (naive chunks, dense) | 0.280 | 0.492 | 0.369 |
| + contextual headers only | 0.400 | 0.672 | 0.510 |
| + section-aware chunking only | 0.307 | 0.453 | 0.370 |
| + drug filter only | 0.378 | 0.623 | 0.485 |
| + hybrid BM25 + RRF only | 0.315 | 0.565 | 0.420 |
| + cross-encoder reranker only | 0.430 | 0.637 | 0.519 |
| headers + reranker | 0.705 | 0.855 | 0.772 |
| **headers + reranker + drug filter (chosen)** | 0.762 | 0.897 | 0.824 |

Without the drug filter, the best combination (headers + BM25 + reranker) still reaches dev recall@5 0.868, so the gain does not depend on the filter. Section-aware chunking *hurt* and hybrid added nothing once the reranker was on, so neither is in the final pipeline. Messy queries (misspellings, brand/generic names) are much harder: see `results/messy_dev.json` (16 scored dev queries: baseline 0.19, headers+reranker 0.375, chosen 0.31 recall@5; small sample).

**Caveat:** eval-v1 and the dev set are **templated** questions built from section text without an LLM (for example "What conditions is Lisinopril used to treat?"). They name the drug exactly and are much cleaner than real user queries, so the recall numbers above are optimistic for real traffic. The messy-query check below is the only signal on realistic wording, and it is small.

Search-only latency (500 queries, 323,280 vectors, p95): **173 ms** (reranker about 103 ms of it); Phase 1 plain dense was 70.5 ms.

Run it:
```bash
python scripts/build_dev_set.py                         # dev questions (already committed)
python scripts/embed_variants.py && python scripts/ablate.py   # local ablation, free (~25 min embedding)
python scripts/build_index_v2.py --config HFR --upsert  # upload the chosen config to Pinecone namespace "v2"
PIPELINE=v2 make up PYTHON=$(which python) && PIPELINE=v2 make up-answer PYTHON=$(which python)
python scripts/eval_phase2.py                           # eval-v1, refuses to run twice
python scripts/bench_latency.py --out results/latency_phase2_search.json --namespace v2
python scripts/eval_generation.py                       # /answer check (needs Bedrock access, `aws login --profile sky`)
python scripts/make_phase2_report.py                    # writes results/phase2.json
```
Generation check (Haiku 4.5, `results/generation_phase2.json`): 15/15 answerable dev questions answered, all with citations that point at retrieved chunks (a valid citation does not prove the answer is faithful; Phase 3 validation covers that); 10/10 unanswerable questions (real drugs missing from the sample) refused with "not enough information". End-to-end latency p50 1.5 s, max 3.1 s (search part p50 about 0.26 s). 41.7K input + 4.2K output tokens, about $0.06.

Answers: `POST :8005/answer` (LangChain retriever over the query service, Claude Haiku 4.5 on Bedrock us-east-2, profile `sky`). Answers cite chunk ids, citations are checked against what was retrieved, and token counts and cost go to `results/bedrock_usage.jsonl` with a hard budget cap.

## Phase 3: extraction, validation and a review queue

A LangGraph workflow reads a drug label and extracts 6 fields, each with a supporting quote. Rules decide which values are auto-approved and which go to a human review queue. Rules, metrics and thresholds were written in `PLAN.md` (M2 pre-registration) before any run.

**Result on 150 `extraction_test` labels (run once, Claude Haiku 4.5; baseline = 100% manual review = 0% auto-approved):**

| Metric | Result |
|---|---|
| Auto-approved, per field (headline) | **69.9%** of 900 field values (target >= 40%) |
| Auto-approved, per label (5 fields; `pharm_class` is always reviewed) | 28.7% |
| Precision of auto-approved values (vs openFDA) | **97.8%**, 95% CI 96.3 to 98.7 (target >= 95%) |
| Field F1 over all values (Phase 4's base-model number) | 0.787 |
| Wrong values sent to review | 89.8% of all wrong values, but only **36%** (8 of 22) once the always-reviewed `pharm_class` is left out |
| Cost | $0.88 for the test run ($1.40 for all of Phase 3, cap $1.50) |

| Field | Auto-approved | Precision of approved | F1 |
|---|---|---|---|
| generic_name | 97% | 0.98 | 0.97 |
| active_ingredients | 97% | 0.97 | 0.97 |
| is_combination | 98% | 1.00 | 1.00 |
| route | 99% | 0.95 | 0.96 |
| has_boxed_warning | 29% | 1.00 | 0.97 |
| pharm_class | 0% (always reviewed) | n/a | 0.11 |

**Nodes:** retrieve (fixed section order, about 3K tokens, no metadata or section names) -> extract (Haiku returns value, verbatim quote and confidence per field) -> validate (schema, quote appears word for word in the source, value inside the quote, cross-field consistency; never sees gold) -> route (auto-approve or queue with reasons) -> summarize (reviewer note for queued items).

**Caveats:**
- The model's self-rated confidence was almost useless (it said "high" on 99% of values for most fields). The verbatim-quote check is the safety gate; confidence can only add a requirement, never override a failed quote check.
- Absence claims (null, empty, false) have nothing to quote, so they are never auto-approved. `pharm_class` is always reviewed because openFDA has no class for about 61% of labels even when the text names one (amendment, 2026-11-06). A second amendment ignores 18 dosage-form words in names. Both were made after the train run and before any test output was seen (see `PLAN.md`); the train numbers behind them are optimistic.
- 14 approved values disagree with openFDA. A manual audit (`results/phase3_audit.json`, one reviewer, not a domain expert) put 9 down to openFDA being incomplete or arguable (empty fields, only the first route scored, `PARENTERAL`), 4 to the same drug worded differently, and 1 to a real model error. The audit does not change the official numbers.
- This is a simulated workload on public labels. `has_boxed_warning` is close to a lookup because a boxed-warning section, if present, is in the text the model reads.

```bash
python scripts/run_extraction.py --split extraction_train --n 100 --summaries 10 --workers 2 --out results/extract_train_raw.jsonl
python scripts/tune_extraction.py      # pre-registered threshold rule on train (free)
python scripts/run_extraction.py --split extraction_test --n 150 --summaries 150 --workers 2 --min-conf medium --out results/extract_test_raw.jsonl   # ONE time
python scripts/eval_phase3.py         # scores the test run, writes results/phase3.json (free)
```

## Phase 4: QLoRA fine-tune of Qwen2.5-1.5B-Instruct

Same extraction task as Phase 3, small open model instead of Haiku. The plan, metrics, hyperparameters and scoring were written in `PLAN.md` (M4 pre-registration) before anything ran. Training ran on a free Kaggle T4 because the SageMaker GPU quota was denied (amendment 2026-11-16, compute only); AWS spend for Phase 4 is $0.

**Result on 150 `extraction_test` labels (each model run once, local mps, bf16, greedy, 512 new tokens, identical prompt; baseline = base Qwen2.5-1.5B-Instruct; `results/phase4.json`):**

| Metric | Base | QLoRA adapter | Change (95% bootstrap CI) |
|---|---|---|---|
| Micro field F1, 6 fields (headline) | 0.627 | **0.817** | +0.190 abs, **+30.3% rel** (+22.6% to +39.3%) |
| Micro field F1, 5 fields (no `pharm_class`) | 0.699 | 0.869 | +0.171 abs, +24.4% rel (+16.9% to +33.1%) |
| Output parsed as JSON | 94.0% (141/150) | 99.3% (149/150) | |
| F1, labels both models parsed (140), 6 fields | 0.646 | 0.816 | +26.4% rel (+18.7% to +34.4%) |
| F1, drugs not among the 1,200 training labels (35), 6 fields | 0.598 | 0.723 | +20.9% rel (+4.0% to +40.4%) |
| F1, same 35 labels, 5 fields | 0.661 | 0.769 | +16.2% rel (-0.4% to +35.4%, includes zero) |
| F1, drugs not in any of the 4,058 train labels (only 10), 6 fields | 0.579 | 0.780 | too few labels for a CI |
| Seconds per label (M4 Pro, mps) | 5.9 | 4.7 | |

Reference: Claude Haiku 4.5 scored 0.787 on the same 6-field F1 in Phase 3 (different prompt path through LangGraph, so context only, not a controlled comparison).

| Field F1 | Base | Adapter |
|---|---|---|
| generic_name | 0.894 | 0.937 |
| active_ingredients | 0.383 | 0.524 |
| is_combination | 0.921 | 0.963 |
| route | 0.869 | 0.849 |
| has_boxed_warning | 0.605 | 0.997 |
| pharm_class | 0.000 | 0.246 |

**Training (Kaggle T4, `results/adapter/train_metrics.json`, `results/train_log.txt`):** 4-bit nf4 QLoRA, fp16, r=16, alpha=32, all 7 projection matrices, lr 2e-4 cosine, batch 1 x 8 accumulation, seed 42. 1,181 of 1,200 examples fit in 4,096 tokens (19 dropped). Loss 0.324 (first 10 steps) -> 0.071 (last 10), validation loss 0.072. torch 2.11.0, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2. The adapter weights (74 MB) are not in git.

**Caveats:**
- **The epoch was cut short.** The pre-registered wall-clock guard (6,000 s) stopped training at step 126 of 148, so the adapter saw about 85% of one epoch. Throughput was about 455 tokens/s, slower than estimated.
- Part of the gain is learning **openFDA's conventions and the output format**, not only reading skill: `has_boxed_warning` (0.60 -> 1.00) and `pharm_class` (0.00 -> 0.25) jump mostly because the model learned what openFDA counts, and parse failures fell from 9 to 1. The parsed-only and 5-field rows are there to separate these effects; the gain stays above +20% relative on both.
- **Drug overlap:** 283 of the 333 distinct generic names in the test split also appear in the train split (different labels, same drug). On the 35 labels whose drug is not among the training labels the gain is smaller (+20.9%) and its 5-field CI includes zero.
- `route` got slightly worse (0.869 -> 0.849). `active_ingredients` is still weak (0.52): the model often lists inactive ingredients.
- Gold labels are openFDA's own fields, which have gaps (for example no `pharm_class` for about 61% of labels even when the text names one), so some "errors" are gold gaps; Phase 3's audit found the same.
- Beating Haiku's 0.787 does not mean a 1.5B model is better than Haiku in general: the adapter was trained on this exact task and label distribution; Haiku was not.

```bash
python scripts/build_train_data.py                                   # 1,200 train + 100 val examples
python scripts/run_qwen_extraction.py --out results/phase4_base_raw.jsonl   # base model, 150 test labels
KAGGLE_USERNAME=<you> python scripts/package_kaggle_dataset.py       # builds the private Kaggle dataset bundle (checks for secrets)
# then follow kaggle/README.md: upload the dataset, run kaggle/train_notebook.ipynb on a T4, download the adapter to results/adapter/
python scripts/run_qwen_extraction.py --adapter results/adapter --out results/phase4_adapter_raw.jsonl   # adapter, same settings as the base run
python scripts/eval_phase4.py                                        # scores both, bootstrap CIs, writes results/phase4.json
```

Unused (kept in case SageMaker is retried): `infra/phase4/` (Terraform, never applied) and `sagemaker/launch_job.py`. `sagemaker/train.py` is the training script used on Kaggle, despite the folder name.
