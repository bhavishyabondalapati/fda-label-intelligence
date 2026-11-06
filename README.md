# FDA Label Intelligence

A GenAI document-processing platform over public FDA drug labels (openFDA, public domain). It chunks, embeds, searches, and extracts structured fields from ~5K prescription drug labels. Every performance number in this repo is produced by a script in this repo, as a relative improvement over a stated baseline.

> **Status: Phase 1 (baseline retrieval served as FastAPI microservices on Pinecone).** Eval set and hit rule are frozen as git tag `eval-v1`.
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
