# FDA Label Intelligence

A GenAI document-processing platform over public FDA drug labels (openFDA, public domain). It chunks, embeds, searches, and extracts structured fields from ~5K prescription drug labels. Every performance number in this repo is produced by a script in this repo, as a relative improvement over a stated baseline.

> **Status: Phase 2 complete (better retrieval + LangChain/Bedrock cited answers).** Eval set and hit rule are frozen as git tag `eval-v1`.
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
