# Phase 7 — Evaluation

## Summary
Scores the Phase 1-6 pipeline against FinanceBench (PatronusAI/financebench, free/public, 150 QA pairs) using two complementary metric families: retrieval metrics (hit rate, MRR, NDCG, precision/recall@k) computed locally from text-overlap relevance judgments, and RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) judged by an LLM + BGE-M3 embeddings. `experiment_tracker.py` appends every run's config + aggregate scores to `data/eval/experiment_log.jsonl` for ablation comparisons. No new pipeline state; this phase only reads Phase 5/6 outputs and writes eval reports.

**2026-06-24 update**: every LLM call in the project (Phase 4 query transform/decomposition, Phase 6 generation, and this phase's RAGAS judge) was migrated from local Ollama to Groq's hosted free-tier API — see "Ollama → Groq Migration" below. Embeddings (BGE-M3) and reranking (BGE-reranker) stay local; only LLM text-generation calls moved off-device.

## Files Changed / Added
- `src/evaluation/models.py` — new — `EvalExample`, `RetrievalEvalResult`, `RagasEvalResult`, `EvalReport` pydantic models
- `src/evaluation/retrieval_metrics.py` — modified (stub → full) — `judge_relevance()` (token-containment heuristic vs FinanceBench evidence text) + `hit_rate_at_k`/`mrr`/`ndcg_at_k`/`precision_at_k`/`recall_at_k`
- `src/evaluation/ragas_evaluator.py` — modified (stub → full) — `RagasEvaluator.evaluate()`: wraps RAGAS's `evaluate()` with a local `ChatOllama` judge LLM and a `_BGERagasEmbeddings` adapter around the already-loaded BGE-M3 model
- `src/evaluation/experiment_tracker.py` — modified (stub → full) — `log_experiment()`/`load_experiments()`: append-only JSONL experiment log
- `scripts/build_eval_dataset.py` — new — filters FinanceBench down to questions whose source filing actually exists in `data/processed/`, writes `data/eval/financebench.json`
- `scripts/run_eval.py` — modified (stub → full) — first full Phase 1-7 end-to-end CLI: per-example query→retrieve→generate→score, aggregate, print, log
- `tests/unit/test_retrieval_metrics.py`, `test_ragas_evaluator.py`, `test_experiment_tracker.py` — new
- `data/raw/MSFT/10-K_2023/`, `data/processed/MSFT/10-K_2023/` — new — ingested so a FinanceBench-overlapping filing exists locally (see Key Design Decisions)
- `data/index/qdrant_local/`, `data/index/bm25_vocab.json` — rebuilt to include MSFT chunks alongside AAPL
- `.claude/CLAUDE.md` — modified — Current Phase bumped to 7

## Key Design Decisions
- **FinanceBench has near-zero overlap with this project's 5-ticker universe (AAPL, MSFT, TSLA, GOOGL, AMZN)**: it covers 84 companies, none of them AAPL/TSLA/GOOGL, and only 1 MSFT question + 3 AMZN questions total (AMZN's are for FY2017/2019, which `SECEdgarLoader.list_filings()` can't reach — see Open Items). Rather than evaluate against zero real questions, MSFT's FY2023 10-K was ingested through Phases 1-3 (download → parse → process → re-index) purely to unlock its 1 matching FinanceBench question. This is a real, if thin, end-to-end signal — not a synthetic test fixture.
- **`build_eval_dataset.py` filters by what's actually on disk, not by a hardcoded ticker list**: it scans `data/processed/*/*/chunks.json` and only keeps FinanceBench rows whose `(ticker, form, year)` matches a real ingested filing. This means the eval set silently grows as more filings get ingested — no code change needed to ask the question "are there now more questions I can evaluate?"
- **Relevance for retrieval metrics is judged by token containment, not exact match**: FinanceBench ships `evidence_text` (a full table/page), not chunk-level ground truth — `judge_relevance()` flags a retrieved chunk relevant if ≥50% of its tokens appear in some evidence text. Containment (chunk-in-evidence), not Jaccard, because the evidence text is much longer than any single chunk and Jaccard would always score low even for a perfect hit.
- **RAGAS judge is `ChatGroq`, not OpenAI**: reuses Phase 6's `cfg.generation.model`/`groq_api_key` as the LLM judge (see "Ollama → Groq Migration" below for why Groq, not local Ollama). `answer_relevancy`'s embedding-similarity step still reuses Phase 3's local `BGEEmbedder` (BGE-M3) via `_BGERagasEmbeddings` — only the judge LLM call is hosted, embeddings never leave the machine.
- **A failed RAGAS metric is `None`, never silently 0.0 or dropped**: `_clean()` converts RAGAS's NaN-on-failure into `None`, and `EvalReport`'s `_mean()` averages only non-`None` scores (and is itself `None`, not `0.0`, if every score in a run failed) — a metric that errored out must not look identical to a metric that scored badly.
- **Experiment log is append-only JSONL, not a single JSON file**: matches CLAUDE.md's "one experiment = one config override" workflow run repeatedly over time; a parse error in one run's row can't corrupt every prior run's record the way a single large JSON array would.

## Execution Flow
One real run via `python scripts/run_eval.py --config-name smoke_test`:
1. `load_config()` resolves `configs/base.yaml`
2. Loads `data/eval/financebench.json` (built ahead of time by `scripts/build_eval_dataset.py`)
3. Per example: `QueryAnalyzer` → `QueryTransformer`/`QueryDecomposer` (if multi-hop) → `RetrievalPipeline.retrieve()` → `Generator.generate()` — identical to `scripts/generate.py`'s Phase 1-6 chain
4. `judge_relevance()` scores the retrieved chunk texts against `EvalExample.evidence_texts` → `retrieval_metrics.py` functions produce one `RetrievalEvalResult`
5. `RagasEvaluator.evaluate()` scores `(question, generated.answer, retrieved chunk texts, ground_truth_answer)` → one `RagasEvalResult`
6. All per-example results aggregate into one `EvalReport` (means skip `None`s)
7. Rich table prints to console; full `EvalReport` written to `data/eval/results/{config_name}.json`; `experiment_tracker.log_experiment()` appends the run to `data/eval/experiment_log.jsonl`

## Data Contract (Input → Output)
- Input (from Phase 5/6, via the live pipeline): `list[RetrievedChunk]`, `GeneratedAnswer`
- Input (eval set): `EvalExample` — FinanceBench question/answer/evidence, built by `scripts/build_eval_dataset.py`
- Output: `EvalReport` — per-example `RetrievalEvalResult`/`RagasEvalResult` rows plus run-level means, persisted to `data/eval/results/*.json` and `data/eval/experiment_log.jsonl`

## Tests
- `test_retrieval_metrics.py` — `judge_relevance` containment threshold (true/false/empty-chunk), `hit_rate_at_k`/`mrr`/`precision_at_k`/`recall_at_k` boundary cases, `ndcg_at_k` perfect/zero/partial ranking
- `test_ragas_evaluator.py` — `_clean()` NaN→None conversion, `RagasEvaluator.evaluate()` wires `SingleTurnSample`/`EvaluationDataset` correctly and extracts all 4 metrics from a mocked `ragas.evaluate()` result (mocked because faithfully faking RAGAS's internal structured-generation protocol for a real `BaseRagasLLM` would reimplement more of RAGAS than the test is worth)
- `test_experiment_tracker.py` — JSONL append (single + multi-run), `load_experiments()` on a missing file
- Full suite: 110 passed (`python -m pytest tests/unit -q`)
- `test_llm_client.py` rewritten for `GroqClient` (was `OllamaClient`) — fakes `groq.Groq().chat.completions.create()`, same monkeypatch pattern as before
- Gap: no test exercises a real Groq API call or a real Qdrant collection — same pattern as every prior phase; covered only by the manual smoke runs in this doc.

## Smoke Test Results (real run, n=1)
`python scripts/run_eval.py --config-name smoke_test` against the one eligible MSFT FinanceBench question ("Has Microsoft increased its debt on balance sheet between FY2023 and the FY2022 period?"):

| Metric | Score |
|---|---|
| Hit Rate@5 | 0.000 |
| MRR | 0.000 |
| NDCG@5 | 0.000 |
| Precision@5 | 0.000 |
| Recall@5 | 0.000 |
| Faithfulness | N/A (judge timed out) |
| Answer Relevancy | 0.9995 |
| Context Precision | N/A (judge timed out) |
| Context Recall | 0.000 |

The run completed end-to-end with no crash — exit 0, report + experiment log both written. The two `N/A`s are RAGAS jobs hitting `RunConfig`'s default 180s timeout against the local 3B model — see "Ollama → Groq Migration" below; this was the proximate trigger for the migration. The 0.0 retrieval/context-recall scores are **not yet root-caused** — could be a genuine retrieval miss on this question, or the 50% token-containment threshold in `judge_relevance()` being too strict against a table-heavy evidence passage (balance sheet figures reformat heavily during chunking). Do not read this single data point as "retrieval is broken" or "retrieval works" — n=1 has no statistical meaning either way.

## Ollama → Groq Migration (2026-06-24)
The local-Ollama RAGAS judge above timed out on 2 of 4 metrics even for a single example — unworkable for any real multi-example run. Every LLM call site in the project (not just the RAGAS judge) was migrated to Groq's hosted free-tier API in one pass: `src/utils/llm_client.py`'s `OllamaClient` became `GroqClient` (same `complete(prompt, system, temperature)` interface, so `src/query/{query_analyzer,query_transformer,query_decomposer}.py` and `src/generation/generator.py` needed only an import + instantiation change), and `src/evaluation/ragas_evaluator.py` swapped `ChatOllama` for `ChatGroq`. Embeddings (`BGEEmbedder`/BGE-M3) and reranking (BGE-reranker) are unchanged — still local, never touch an API.

**Re-ran the same n=1 smoke test on Groq** (`python scripts/run_eval.py --config-name groq_smoke_test`):

| Metric | Ollama run | Groq run |
|---|---|---|
| Hit Rate@5 | 0.000 | 0.000 |
| MRR | 0.000 | 0.000 |
| NDCG@5 | 0.000 | 0.000 |
| Precision@5 | 0.000 | 0.000 |
| Recall@5 | 0.000 | 0.000 |
| Faithfulness | N/A (timeout) | 0.800 |
| Answer Relevancy | 0.9995 | **0.000** |
| Context Precision | N/A (timeout) | 0.500 |
| Context Recall | 0.000 | 0.000 |

The judge-timeout problem is gone — all 4 RAGAS metrics now return real scores on the first try, no `RunConfig` tuning needed. Total wall time dropped from ~11 min to ~4.5 min, and nearly all of that remaining time is local BGE-M3/reranker model loading + dense-embedding batches for the 5 query variants (HyDE + multi-query), not LLM latency — the LLM calls themselves (intent classification, HyDE, multi-query, generation, 4 judge calls) are now seconds, not minutes, confirmed separately via `scripts/generate.py` (full Phase 1-6 run completed in ~1.5 min, almost entirely local model load time).

Retrieval metrics are unchanged (0.0, still unresolved — same open question as before, not a regression). **New, also unresolved**: `answer_relevancy` flipped from ~1.0 to exactly 0.0. `EvalReport`/`RagasEvalResult` don't persist the actual generated answer text, so this wasn't root-caused before writing this up — could be a genuinely different (worse) answer from the Groq-hosted model vs. local llama3.2:3b, or an issue in how `answer_relevancy` scores against the `_BGERagasEmbeddings` adapter. Treat as an open question, not a regression verdict — n=1 still has no statistical meaning.

## Config Keys Used
- `configs/base.yaml` → `evaluation.dataset`, `evaluation.metrics`, `evaluation.batch_size` (batch_size not yet wired — see Open Items)
- Reuses `generation.model`/`generation.groq_api_key` (RAGAS judge), `embeddings.dense_model`/`embeddings.batch_size` (RAGAS embeddings), `retrieval.top_k_rerank` (metric @k cutoff)

## Open Items / Deferred
- **FinanceBench corpus overlap is fundamentally thin and not fixable from Phase 7 alone**: only 1 MSFT + 3 AMZN questions exist for this project's 5-ticker universe across all of FinanceBench's 150 questions, and zero for AAPL/TSLA/GOOGL. The 3 AMZN questions (FY2017, FY2019) are unreachable because `SECEdgarLoader.list_filings()` (Phase 1) only reads EDGAR's `filings.recent` array, which covers roughly the last ~1000 filings per company — too recent to include Amazon's 2018/2020 filing dates. Extending it to paginate `filings.files[]` for older filings is a Phase 1 change, not Phase 7's to make.
- ~~RAGAS judge timeouts at the default 180s `RunConfig`~~ — resolved by the Groq migration above; no longer an open item.
- **`evaluation.batch_size` in `configs/base.yaml` is declared but unused**: `run_eval.py` evaluates one example at a time, sequentially. Worth revisiting once example counts are large enough that batching the RAGAS judge calls matters for runtime.
- **The 0.0 retrieval-metric result is unexplained across both runs** (see Smoke Test Results / migration table above) — needs a real multi-example run (once more filings are ingested) before drawing any conclusion about retrieval quality.
- **The Groq-run `answer_relevancy` collapsing to 0.0 (from ~1.0 on Ollama) is unexplained** — `EvalReport` doesn't persist generated answer text, so this couldn't be root-caused without a re-run; worth adding the answer text to `RagasEvalResult` or the report for future debugging.
- **CLAUDE.md's Experiment Log table is intentionally left with placeholder dashes**, not the n=1 smoke numbers — one example is not a baseline worth committing to that table; populate it once a real-sized eval run exists.
