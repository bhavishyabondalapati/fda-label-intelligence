# FDA Label Intelligence

A GenAI document-processing platform over public FDA drug labels (openFDA, public domain). It chunks, embeds, searches, and extracts structured fields from ~5K prescription drug labels. Every performance number in this repo is produced by a script in this repo, as a relative improvement over a stated baseline.

> **Status: Phase 0 (data, eval set, baseline).** See [PLAN.md](PLAN.md) for the full roadmap and exact metric definitions. This README is a stub and is completed in Phase 5.

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
