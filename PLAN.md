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

#### M2 pre-registration (written 2026-11-06, BEFORE any Phase 3 run; frozen before the train run)

**Review unit.** Headline = **per field**: every extracted field value is approved or flagged on its own, and a reviewer only checks flagged fields. `auto-approval rate (per field) = approved field values / all field values`. Secondary = **per label**: a label is auto-approved only if all 6 of its fields are approved. Baseline for both: 0% (everything reviewed).

**Fields (6).** `generic_name` (openfda.generic_name), `active_ingredients` (list, openfda.substance_name), `is_combination` (bool, derived: 2+ distinct ingredients after salt stripping), `route` (openfda.route, first listed), `has_boxed_warning` (bool, a boxed_warning section exists), `pharm_class` (list, openfda.pharm_class_epc; empty list = none). The model only sees section text; the openfda block, label metadata and section names are never in the prompt.

**Matching rules** (implemented and tested in `src/extract/scoring.py`; changing them after the train run is an amendment):
- Normalize: lowercase, every non-alphanumeric character becomes a space, whitespace collapsed.
- Salt/form words are removed from BOTH sides before comparing names (frozen list `SALT_WORDS` in the module: hydrochloride, hcl, sodium, potassium, sulfate, tartrate, citrate, etc.). If removal would leave nothing (e.g. "sodium chloride"), the name is kept whole. So "SILDENAFIL CITRATE" matches "sildenafil"; "metoprolol succinate" matches "metoprolol tartrate" (known, accepted looseness: a reviewer sees the same drug). A strict (no salt stripping) accuracy is also reported as a secondary number.
- `generic_name`: correct if the sets of remaining tokens are equal (word order and "and"/commas ignored).
- `active_ingredients`: correct if the SETS of salt-stripped ingredient names are equal. For F1, each ingredient is one item.
- `is_combination`: gold = (number of distinct salt-stripped gold ingredients >= 2). Exact boolean match.
- `route`: uppercase, normalized; exact match to the gold route. Gold route empty = expected null. Only the first openFDA route is gold (known limitation: multi-route labels are scored against one route).
- `has_boxed_warning`: exact boolean match.
- `pharm_class`: each class has a trailing bracket suffix such as `[EPC]` removed, then is normalized; correct if the SETS are equal. Null and the empty list are the same thing ("none").
- **Field F1** is micro-averaged over all field values: scalar fields count as one item (match = TP; wrong value = FP and FN; null prediction with non-null gold = FN; non-null prediction with null gold = FP; both null = ignored); list fields count each element.

**Validation (never sees gold).** A field is auto-approved only if ALL gates pass:
1. Schema valid (right types, route from the allowed list).
2. **Verbatim-quote gate (required):** the field has at least one supporting quote (3 to 300 characters) that appears word for word in the source text sent to the model (case-insensitive, whitespace collapsed).
3. Value-support gate: the value's own tokens (or, for route, a cue word for that route) appear inside the quote(s).
4. Cross-field consistency: `is_combination` must equal (2+ ingredients in `active_ingredients`) and is approved only if `active_ingredients` is approved.
   Also: `generic_name` and `active_ingredients` must name the same drug (one set of salt-stripped words contained in the other), otherwise both are flagged. Quote-support details (route cue words, the 'warning' cue for boxed warnings, a 12,000 character / about 3K token section cap in a fixed section order) are frozen in `src/extract/validate.py` and `src/extract/retrieve.py` before the train run.
5. Confidence gate (below).
- **Absence is never auto-approved.** A null, empty list, or `false` claim has no text to quote, so it is always flagged (this includes `has_boxed_warning = false` and `pharm_class = []`). `is_combination` is the one derived exception (rule 4).
- **Confidence signal.** Claude on Bedrock gives no token probabilities, so confidence is the model's own rating (`high`/`medium`/`low`) per field. That is weak evidence, so it can only ADD a requirement (a minimum level) and can never rescue a field that failed gates 1 to 4.
- **Threshold choice (train only).** On 100 `extraction_train` labels, compare three settings: no confidence requirement, `medium` or above, `high` only. Choose the setting with the highest per-field auto-approval rate whose train precision (approved fields that match gold) is >= 96% (the 1-point margin covers train-to-test drift). If none qualifies, the most strict setting is used and the shortfall is reported.

**Amendments after the train run (2026-11-06; made BEFORE any `extraction_test` output was generated or viewed).**
Train evidence (100 labels, 600 field values) under the frozen rules: 76.7% of fields auto-approved, but precision of approved values only 85.2%, so the 95% target was missed. Confidence was no help (the model said "high" for 99% of values), so the rule picked `high`.
1. *`pharm_class` is always reviewed.* 44 of the roughly 65 wrong approved values were `pharm_class`; in 36 of those 44 the openFDA record has no class although the label text names one. The answer key is incomplete for about 61% of labels, so the field cannot be auto-approved reliably. It stays in the extraction task (and in Phase 4) and is still scored for accuracy and F1. Per-label approval is now computed over the other 5 fields (per-label over all 6 would be 0 by construction).
2. *Dosage-form words are ignored in names, on both sides* (gold and prediction), next to the salt words. The exact list (`FORM_WORDS` in `src/extract/scoring.py`): `cream gel ointment solution injection ophthalmic oral topical er xr extended release for suspension capsule capsules lotion spray`. If removal would leave nothing, the name is kept whole. Reason: "hydrocortisone cream" vs "hydrocortisone" is a formatting difference. This only affects `generic_name` and `active_ingredients` matching (and the value-support check).
Re-scoring the same saved train outputs under both amendments (no new model calls): 67.0% of fields auto-approved, precision 97.8% (95% CI 95.8 to 98.8), 92% of wrong values sent to review. These train numbers are optimistic because the amendments were chosen on the same 100 labels. Applying the frozen threshold rule to the amended train scores picks `medium` (identical to no requirement except for 4 low-confidence route values; `high` is 0.7 points lower in approval at the same precision), so `medium` is the test setting, and `high` is reported next to it.

**Test run.** Once, on 150 labels of `extraction_test` (a fixed, seeded subset chosen before the run), with the chosen setting. Reported: per-field and per-label auto-approval rate, precision of auto-approved fields (target >= 95%), field F1 over all items (Phase 4's base-model number), and the queue catch rate (wrong extractions that were flagged / all wrong extractions). The simulated-workload caveat applies. Cost cap: $1.50 on a separate ledger.

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

## M4 pre-registration: QLoRA fine-tune of Qwen2.5-1.5B-Instruct (written 2026-11-06, before any Phase 4 data was built or any model run)

**Goal.** Same extraction task as Phase 3 (6 fields; value + code-built quote + confidence JSON). Compare the BASE Qwen2.5-1.5B-Instruct against the same model with a QLoRA adapter on the SAME 150 `extraction_test` labels (the Phase 3 sample: first 150 by sha256 of `phase3-extraction_test:<set_id>`). Haiku 4.5's 0.787 is shown only as a reference.

**Training data.**
- Source: `extraction_train` only. 1,200 labels for training (first 1,200 by sha256 of `phase4-train:<set_id>`) and the next 100 for validation loss monitoring only (no tuning of anything). Test labels and `retrieval_eval` are never read when building data.
- Input: byte-for-byte the Phase 3 prompt (`EXTRACT_SYSTEM` + the same `select_text` section text). No openFDA block, metadata or section names.
- Targets: openFDA gold values in the Phase 3 forms (lowercase names, route from the allowed list, a route outside the list becomes `OTHER`), confidence always "high".
- **Quotes are built by code**, never by a model: the gold value is searched (case and whitespace insensitive) in the source text, and a window of 3 to 300 characters around the first match, snapped to word boundaries, is used and checked with the existing `quote_ok`. Route uses the Phase 3 route cue words; `has_boxed_warning=true` quotes the start of the boxed warning; `false`, empty lists and `is_combination` get no quote. If the value cannot be found in the text the field is kept with `quotes: []`. Examples are NOT dropped for that (dropping would favour easy labels). The share of fields with a real quote is reported.
- **`pharm_class` loss masking:** the tokens of the `pharm_class` value and its quotes are excluded from the loss whenever the gold list is empty, or non-empty but no class phrase is found in the text. Reason: openFDA has no class for about 61% of labels even when the text names one, and training on those would teach "output []". Scoring on test still uses the gold as is.
- Examples longer than 4,096 tokens are dropped and counted (should be rare, since the section text is capped at 12,000 characters).

**Training (fixed, no search).** 4-bit nf4 quantization, **fp16** compute (a T4 has no good bf16), LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; learning rate 2e-4, cosine, 3% warmup, 1 epoch, micro-batch 1 with 8 accumulation steps (effective 8, about 150 optimizer steps), seed 42, gradient checkpointing, loss on answer tokens only. A NaN loss aborts the job early. A wall-clock guard saves the adapter after 6,000 s of training; if it ever fires, the README says the epoch was cut short.

**Instance and cost guard (price verified 2026-11-06 with the AWS Price List API, region us-east-2, usage type `USE2-Train:ml.g4dn.xlarge`: $0.736 per hour).**
- `ml.g4dn.xlarge` (NVIDIA T4 16 GB), **on-demand** (no managed spot), instance count 1. Quota request is still pending, so nothing is launched.
- `max_run` = 6,600 s, a ceiling of 1.83 h x $0.736 = $1.35. Allowing up to 10 more minutes of billed start-up and download, the worst case is $1.47, under the $1.50 limit.
- The label count (1,200, about 3.2M tokens) was chosen so that even a pessimistic 500 tokens/s finishes in about 108 minutes. My expected rate is 800 to 1,500 tokens/s (an estimate, NOT measured, since the local smoke test runs on mps), which means 36 to 68 minutes and **$0.45 to $0.85**. The job prints its measured tokens/s after 10 steps.

**Evaluation (identical for base and adapter).** Run locally on mps in bf16, greedy decoding, up to 512 new tokens, same prompt. Output that is not a JSON object counts as all fields wrong. Scoring is `src/extract/scoring.py` (frozen in Phase 3). Reported for each model, as absolute and relative improvement of adapter over base, each with a 95% bootstrap CI over labels (2,000 resamples, seed 42):
1. **Headline: micro field F1, all 150 labels**, with all 6 fields AND with 5 fields (`pharm_class` left out).
2. **Parsed-only F1** (format vs skill): F1 over the labels whose output parsed as JSON, for each model, plus the labels where BOTH parsed. Also the parse rate of each model.
3. **Unseen-drug F1** (memorized vs generalized): F1 on the test labels whose `generic_name` key (`name_key`) does NOT occur among the 1,200 labels the adapter was trained on. The same figure against all 4,058 train labels is also reported. The sizes of these subsets are stated next to the numbers.
- Target: at least +15% relative micro F1 (6 fields, all labels). The resume bullet will use the number that comes out, whether or not it reaches 15%.
- The test set is evaluated once per model. Nothing about the adapter is chosen from test results.

**Caveats to print with the result.** 283 of the 333 distinct test generic names also appear in the full train split (different labels, same drugs); part of the gain is learning openFDA's conventions (first route only, salt stripping), which is why the unseen-drug number is shown; the adapter is compared with a small base model, not with Haiku.

**Addendum 2026-11-06 (scoring of malformed values; written before the base-model test run, applied equally to base and adapter).** A model output counts as *parsed* if it is a JSON object (code fences are stripped). Inside a parsed object, a list field (`active_ingredients`, `pharm_class`) given as a single string is read as a one-element list (otherwise the string would be scored letter by letter); a missing or null field is read as absent; booleans must be real booleans or the field counts as absent. This lives in `src/finetune/metrics.py`; the Phase 3 `scoring.py` is unchanged.
