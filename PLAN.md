# PLAN

Goal: a repo whose numbers honestly back three claims. Every number comes from a script in this repo and is written to `results/*.json`.

| # | Claim | What the repo must show |
|---|-------|-------------------------|
| 1 | Cut manual document review by 40% and improved retrieval accuracy by 30% | Review reduction and recall@5 gain (definitions below) |
| 2 | Sub-3-second semantic search over 100K+ embeddings | p95 latency < 3 s with >= 100,000 vectors in the index |
| 3 | 15% better domain extraction than base models via LoRA/QLoRA/PEFT on SageMaker | Field-level F1 gain of adapter over base model |

**Rules for every result:** the baseline is stated before the improved system is run; metrics are defined here, before results exist; if a target is missed, the repo reports the real number and the README wording changes, not the metric.

## Fixed constraints

- Data: openFDA drug label bulk JSON, human prescription labels only, ~5,000 labels (seed 42), enough for 100K+ chunks.
- AWS: profile `sky`, region `us-east-2` only, total spend under $5, no always-on endpoints. No AWS calls in Phase 0.
- Secrets only in `.env` (gitignored). `.env.example` holds placeholders.
- Pre-registration: gold data, splits, and metrics are fixed before models are scored.

## Data splits (fixed in Phase 0, by hash of `set_id` + seed)

| Split | Share | Used for |
|-------|-------|----------|
| `retrieval_eval` | ~10% | Source labels for the ~200 gold retrieval questions |
| `extraction_test` | ~10% | Held-out labels for extraction F1 (never trained on) |
| `extraction_train` | ~80% | Fine-tuning data in Phase 4 |

The **search index contains all ~5,000 labels** (needed for scale and realism). Gold questions come only from `retrieval_eval` labels, and fine-tuning examples come only from `extraction_train`, so eval data never leaks into training.

## Metric definitions

### M1. Retrieval accuracy = recall@5
- Eval set: ~200 gold questions, each with one gold `(set_id, section)`. Built by `scripts/build_eval_set.py` from section text of `retrieval_eval` labels, balanced across section types. Questions use templates (no LLM).
- Relevance (amended 2026-11-05, see note below): a retrieved chunk is **relevant** if it comes from **any label with the same normalized drug name and the same route** as the gold label, **and** it covers that label's own copy of the gold section (overlap of at least `min(200, section length)` characters for naive chunks; section id match for section-aware chunks). A question is a **hit** if any of the top 5 chunks is relevant.
- Questions name only the drug, not the manufacturer. For a drug name that appears with more than one route in the corpus, the route is added to the question text, e.g. "Lidocaine (transdermal)", so the question says which one it means.
- The **strict** variant (exact gold `set_id`) is kept as a secondary cross-check in `results/*.json` under `strict_crosscheck`.
- Reported metrics: recall@1, recall@5 (primary), MRR@10, always as absolute values and as relative improvement over the baseline.
- `recall@5 = hits / questions`; `MRR@10` = mean of 1/rank of the first relevant chunk in the top 10 (0 if none).
- Reported as **relative improvement**: `(recall_new - recall_baseline) / recall_baseline`. Target: >= 30% relative. Percentage points are reported alongside, never instead.
- Baseline (Phase 0): naive fixed-size chunks (1000 chars, 100 overlap, section boundaries ignored), `all-MiniLM-L6-v2`, dense only, local numpy index.
- Known limitation: templated questions are cleaner than real user queries. If budget allows, Phase 2 adds a second, LLM-written question set and reports both.

### M2. Review reduction = auto-approval rate at precision >= 95%
- Each extracted field value gets a confidence score. A threshold chooses which extractions are **auto-approved**; the rest go to the human review queue.
- `auto-approved precision` = share of auto-approved extractions that match gold. The threshold is chosen on `extraction_train` labels (never on test) as the lowest one giving precision >= 95%.
- On `extraction_test`: `review reduction = auto-approved extractions / all extractions`, reported only if test precision of the auto-approved set is also >= 95%.
- Baseline: 100% manual review (every extraction reviewed). So a 40% review reduction means 40% auto-approved. This is a **simulated** workload on public labels, and the README says so.

### M3. Latency = p95 over 500 queries
- 500 distinct queries against the live query service with >= 100,000 vectors in Pinecone, measured client-side end to end (HTTP request in, results out, including query embedding). Warm-up queries (20) are excluded and stated.
- Reported: p50, p95, p99, vector count, machine, and network location.
- This claim is an absolute threshold (p95 < 3.0 s), so there is no relative improvement to report. Latency of the local brute-force baseline is reported only for context.

### M4. Extraction accuracy = field-level F1
- Task: given label text, output JSON with a fixed field set. Gold values come from openFDA structured data; the exact field list is frozen at the start of Phase 4, before any model output is seen.
- Matching: normalized exact match per field (lowercase, whitespace, punctuation stripped). Micro-averaged F1 over all fields on `extraction_test`.
- Reported as **relative improvement**: `(F1_adapter - F1_base) / F1_base`. Base = Qwen2.5-1.5B-Instruct with the same prompt, no adapter. Target: >= 15% relative.

## Amendments

**2026-11-05: relevance rule for M1 changed (before any Phase 1+ result existed).**
- *What changed:* the first baseline run counted a chunk as a hit only if it came from the exact gold `set_id`. The rule is now "same normalized drug name + same route + gold section" (see M1), the manufacturer was removed from the question text, and the strict rule became a cross-check.
- *Why:* a sanity check of sample questions showed the strict rule marked correct answers wrong. Many generic drugs have near-identical labels from several manufacturers, so a copy from another manufacturer gives the same answer to "what are the contraindications of X?". The strict eval was penalizing the retriever for picking an identical copy, and 152 of 200 questions had to name the manufacturer to be answerable, which no real user does.
- *Why this is legitimate:* the flaw was found by inspecting baseline output, before any improved system was built. The baseline is re-run under the new rule, so baseline and every later result use the same definition. The rule will not be changed again after Phase 1+ results exist.
- Route is included because labels for the same drug differ by route (e.g. lidocaine injection vs patch have different warnings).
- Note for Phase 2: adding the drug name and section heading to each chunk's text ("contextual chunk headers") is a standard retrieval technique and is allowed. It will be named as such and measured on its own.

**2026-11-05 (second amendment, same day): multi-route drugs kept; eval FROZEN as `eval-v1`.**
- The first fix skipped drug names that come in more than one route (e.g. lidocaine injection vs patch). That quietly removed the harder, more common drugs from the eval. They are back, with the route in the question text (34 of 200 questions).
- The eval set (`data/eval/gold_questions.jsonl`) and the hit rule (`src/eval/metrics.py`) are now frozen under git tag `eval-v1`. `data/eval/eval-v1.lock` pins their SHA-256 hashes and `tests/test_eval_frozen.py` fails if either changes. Every later phase is measured against `eval-v1`.
- Frozen baseline (`results/baseline.json`): **recall@1 = 0.230, recall@5 = 0.475 (95/200), MRR@10 = 0.331**. Strict cross-check: recall@5 = 0.235.
- Phase 2 target for M1: 30% relative improvement over 0.475, i.e. recall@5 >= 0.6175, which means at least 124 of 200 questions (0.62).

## Phases

### Phase 0: data, eval, baselines (this session)
- Scaffold, `PLAN.md`.
- `scripts/download_subset.py`: download bulk files, keep human Rx labels, deterministic 5K sample, parse into per-section records with metadata, assign splits.
- `scripts/build_eval_set.py`: ~200 gold questions to `data/eval/gold_questions.jsonl`.
- `scripts/run_baseline.py`: naive chunking + MiniLM + dense numpy index, recall@5 to `results/baseline.json`.
- Exit: baseline numbers shown and reviewed.

### Phase 1: FastAPI microservices + Pinecone + latency
- Same retrieval as the baseline (naive chunks, MiniLM, dense only), served as services. No retrieval improvements yet.
- Services: `ingest`, `chunk`, `embed`, `query`. Vectors in Pinecone serverless (free tier), >= 100K. Only small metadata lives in Pinecone; chunk text is looked up locally by `chunk_id`.
- Parity check: eval-v1 through the query service must match the local exact baseline (recall@1/@5, MRR@10); any gap is explained (approximate vs exact search).
- `scripts/bench_latency.py`: 500 queries, p95 (M3).
- Exit: p95 reported against the 3 s threshold.

### Phase 2: LangChain RAG + Bedrock + retrieval improvements
- Bedrock (Claude Haiku class) for answer generation, a few cents total.
- Retrieval improvements, each measured separately against the Phase 0 baseline: section-aware chunks with drug/section headers, hybrid dense + BM25, reranking.
- Exit: recall@5 relative improvement (M1) per technique and combined.

### Phase 3: LangGraph retrieve -> validate -> summarize + review queue
- Graph: retrieve, extract fields, validate (check values appear in source text), summarize; low-confidence items go to a review queue.
- Exit: review reduction (M2) on `extraction_test`.

### Phase 4: QLoRA/PEFT fine-tune on SageMaker
- Fine-tune Qwen2.5-1.5B-Instruct with QLoRA on `extraction_train` using one SageMaker training job (single GPU, spot if possible, a few dollars at most). Base vs adapter evaluated on `extraction_test` (M4); evaluation runs locally on the Mac or inside the job, never on a standing endpoint.
- Risk: new AWS accounts often have 0 quota for GPU training instances, so request the quota early. Cost cap is enforced by a hard `max_run` on the job.
- Exit: F1 relative improvement (M4).

### Phase 5: Terraform + Makefile + README
- `infra/` with Terraform, `make deploy` / `make destroy`, nothing left running.
- Final README with architecture, results table (every number linked to its `results/*.json`), and honest caveats.

## Budget (AWS, us-east-2)

| Item | Estimate |
|------|----------|
| Embeddings | $0: local MiniLM on the Mac |
| Bedrock generation (Haiku class, a few hundred calls) | < $1 |
| SageMaker QLoRA job (1 GPU, ~1 h, spot if possible) | ~$1.5 to $3 |
| S3, misc | < $0.50 |
| **Total** | **< $5** |
