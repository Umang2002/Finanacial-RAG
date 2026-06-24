# Financial RAG — Project Context for Claude Code

## What This Project Is
End-to-end Hybrid Search RAG system for SEC financial document intelligence.
Built for learning (understand every component) and resume (production-grade patterns).
**Free stack — embeddings, reranking, and the vector DB are all local; LLM calls (query
transform, generation, RAGAS judge) run on Groq's free-tier hosted API (see LLM Provider
Migration Notes).**

## Architecture Summary
7-phase pipeline:
1. Ingestion → SEC EDGAR API (free), PyMuPDF/pdfplumber for PDFs, BeautifulSoup for HTML
2. Processing → cleaning, recursive/semantic/sentence-window/parent-child chunking, table serialization
3. Indexing → dense (BGE-M3 via sentence-transformers → Qdrant) + sparse (BM25 → Qdrant sparse vectors)
4. Query → intent classification, HyDE, multi-query expansion, decomposition for multi-hop
5. Retrieval → dense ANN + BM25 keyword, RRF fusion, cross-encoder reranking (BGE-reranker, local)
6. Generation → context assembly, citation-aware prompting, structured output (answer + citations)
7. Evaluation → RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) + retrieval metrics (MRR, NDCG, Hit Rate)

## Key Tech Choices & Why
- **Qdrant**: supports both dense vectors AND sparse vectors in one collection — no need for separate DBs
- **BGE-M3** (BAAI/bge-m3): free, local embedding model via sentence-transformers; dense dim = 1024
- **BGE-reranker** (BAAI/bge-reranker-v2-m3): free cross-encoder reranker, runs locally via sentence-transformers
- **Groq**: free-tier hosted LLM API (OpenAI-compatible); fast inference, no local model loading. Replaced an earlier local-Ollama setup — see LLM Provider Migration Notes.
- **RRF (Reciprocal Rank Fusion)**: simple, parameter-free fusion — outperforms weighted sum in practice
- **RAGAS**: LLM-as-judge eval framework — measures RAG quality without human labels
- **FinanceBench**: open QA benchmark over real SEC filings — gives credible, comparable eval numbers

## Free Stack
| Component | Tool | Cost |
|---|---|---|
| Embeddings | BGE-M3 (sentence-transformers) | Free, local |
| LLM | Groq API (llama-3.3-70b-versatile) | Free tier, hosted |
| Reranker | BGE-reranker-v2-m3 | Free, local |
| Vector DB | Qdrant (embedded local-mode, no Docker) | Free, local |
| Ingestion | SEC EDGAR API | Free, public |
| Observability | rich console logging | Free |

## Dataset
- Source: SEC EDGAR (free, public API) — 10-K and 10-Q filings
- Tickers: AAPL, MSFT, TSLA, GOOGL, AMZN (years 2021-2024)
- Eval: FinanceBench (150+ QA pairs with ground-truth answers)

## Dev Rules (always follow)
- Every function must have a docstring explaining WHAT it does and WHY (not just what)
- Add a `# LEARN:` comment wherever a non-obvious design choice was made
- Type hints on every function signature
- Config comes from `configs/base.yaml` via Hydra — no hardcoded values in src/
- Use `pydantic` models for all data structures that cross module boundaries
- Log with `rich` for all output — no plain print()
- One experiment = one config file override in `configs/experiments/`

## Prerequisites (one-time setup)
```bash
# 1. Get a free Groq API key → https://console.groq.com/keys
#    Put it in .env as GROQ_API_KEY=gsk_... (never commit it — .env is gitignored,
#    configs/base.yaml only references it via ${oc.env:GROQ_API_KEY})
# 2. BGE-M3 + BGE-reranker download automatically on first use (~2.3GB + reranker)
# Qdrant runs embedded (no Docker / no server) — see "Vector DB" row above.
# docker-compose.yml is kept for an optional server-mode swap later
# (QdrantClient(path=...) -> QdrantClient(url="localhost")) if Docker becomes available.
```

## Running the Project
```bash
# Install deps
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Download filings
python scripts/download_filings.py --ticker AAPL --years 2022 2023

# Parse + chunk
python scripts/parse_filings.py --ticker AAPL --years 2023
python scripts/process_filings.py --ticker AAPL --years 2023

# Build hybrid index (dense BGE-M3 + sparse BM25, embedded Qdrant)
python scripts/build_index.py --ticker AAPL --years 2023

# Run evaluation
python scripts/run_eval.py --config configs/base.yaml
```

## Current Phase
Phase 8 — Frontend + Deployment (chat + citations, retrieval debug panel,
eval dashboard, and visual restyle all built and verified end-to-end;
deployment is the only remaining item — see Phase 8 Notes)

## Phase 6 Notes
- `Generator` wires `ContextAssembler` + `GroqClient` (was `OllamaClient`
  until the 2026-06-24 migration — see LLM Provider Migration Notes) +
  `output_parser` into one `generate(query, chunks)` call returning a
  `GeneratedAnswer`.
- `ContextAssembler` assigns citation numbers in relevance order (1 =
  best) but physically places chunk text in sandwich order (best chunks
  at both ends, weakest in the middle) — mitigates lost-in-middle while
  keeping citation numbers meaningful to the model.
- Citation style is config-gated (`generation.citation_style`) but only
  `inline` is implemented; `Generator.__init__` raises on any other value
  rather than silently producing unparsable output.
- `scripts/generate.py` is the first true end-to-end CLI: intent
  classification -> HyDE/multi-query -> decomposition (multi-hop only) ->
  retrieval -> generation -> printed answer + citations table.

## Phase 7 Notes
- `RagasEvaluator` judges with `ChatGroq` (reuses `generation.model`/
  `groq_api_key`) + a BGE-M3 embeddings adapter (reuses
  `embeddings.dense_model`, stays local) — see LLM Provider Migration Notes
  for why this is Groq and not local Ollama as originally built.
- `retrieval_metrics.judge_relevance()` flags a retrieved chunk relevant if
  ≥50% of its tokens are contained in FinanceBench's `evidence_text` — there's
  no chunk-level ground truth, only page-level evidence, so containment
  (not Jaccard) is the only sane comparison.
- FinanceBench has almost no overlap with this project's ticker universe:
  1 MSFT + 3 AMZN questions total, 0 for AAPL/TSLA/GOOGL, out of 150
  questions. MSFT's FY2023 10-K was ingested (Phases 1-3) just to unlock its
  1 question; the AMZN ones are unreachable until Phase 1's
  `SECEdgarLoader.list_filings()` paginates EDGAR's older-filings index.
  See `docs/phases/phase7_evaluation.md` Open Items.
- Smoke-tested end-to-end on that 1 MSFT question, twice (once on local
  Ollama, once after migrating to Groq) — both runs ran clean (exit 0).
  Retrieval metrics scored 0.0 on both — not yet root-caused (real miss vs.
  threshold), and n=1 isn't enough to conclude anything either way. See
  full before/after table in `docs/phases/phase7_evaluation.md`.

## LLM Provider Migration Notes (2026-06-24)
- The local-Ollama RAGAS judge (Phase 7) timed out on 2 of 4 metrics on a
  single example — the local 3B model couldn't keep up even for a smoke
  test. Migrated every LLM call site in the project — not just the judge —
  from local Ollama to Groq's hosted free-tier API in one pass.
- `src/utils/llm_client.py`: `OllamaClient` → `GroqClient`, same
  `complete(prompt, system, temperature)` interface, so
  `src/query/{query_analyzer,query_transformer,query_decomposer}.py` and
  `src/generation/generator.py` only needed an import + instantiation swap
  (`host=...` → `api_key=...`). `src/evaluation/ragas_evaluator.py` swapped
  `ChatOllama` for `ChatGroq`.
  `pip install groq langchain-groq` pulled `langchain-core>=1.0`, which
  breaks `langchain`/`langchain-community`/`langchain-openai` (all pinned
  `<1.0`, needed for `ragas` imports) — pinned `langchain-groq==0.2.5`
  instead, which depends on the compatible `langchain-core<1.0` line.
- `configs/base.yaml` `generation.provider` is now `groq`, model is
  `llama-3.3-70b-versatile`, key comes from `${oc.env:GROQ_API_KEY}`
  (`.env`, gitignored — never the literal key in `configs/base.yaml`,
  which is git-tracked).
- Embeddings (BGE-M3) and reranking (BGE-reranker) are unchanged — still
  local sentence-transformers models, never touch an API.
- Result: RAGAS judge timeouts disappeared (all 4 metrics return real
  scores now), and `scripts/generate.py` end-to-end runtime dropped from
  several minutes to ~1.5 min (almost entirely local model load time now,
  not LLM latency).
- **Caution for future sessions**: a real Groq key was once pasted directly
  into `configs/base.yaml` (git-tracked) instead of `.env` — caught before
  commit and moved to `.env`; recommended rotating it since it briefly
  touched a chat transcript. Always put API keys in `.env` only; reference
  via `${oc.env:VAR_NAME}` in config files.

## Phase 8 — Frontend + Deployment (in progress)
- Next.js frontend, deployed (resume-facing, public link) — deployment still pending.
- Backend API layer: FastAPI wrapping RetrievalPipeline + generation (`/query`, `/health`). Chosen over Next.js-routes-calling-Python.
- Frontend v1 scope: chat-style query input + streamed answer w/ citations, retrieval debug panel (dense/sparse/hybrid/reranked per-stage view, like `scripts/retrieve.py`), eval dashboard (RAGAS + retrieval metrics).
- Deployment: backend already runs on Groq (see LLM Provider Migration Notes) — no separate local/prod LLM split needed anymore, one fewer thing to configure for deployment.

### Phase 8 Notes
- v1 (chat + citations) shipped first; this pass added the retrieval
  debug panel + eval dashboard that v1 deferred, plus a full visual
  restyle. Deployment is still the only thing left in Phase 8.
- `src/api/main.py`: FastAPI app, `lifespan` builds every Phase 4-6
  component (`QueryAnalyzer`, `QueryTransformer`, `QueryDecomposer`,
  `RetrievalPipeline`, `Generator`) once at startup and stashes them on
  `app.state` — model loads (BGE-M3 + BGE-reranker) dominate
  `scripts/generate.py`'s ~1.5min runtime (see Phase 6/7 notes), so paying
  that cost per-request would make every query slow. `POST /query` runs
  the same intent -> transform -> decompose -> retrieve -> generate
  sequence `scripts/generate.py` uses.
  - `QueryRequest.debug: bool` opts into `_run_debug_stages()`, which
    replays dense/sparse/hybrid(RRF)/reranked over the raw query alone —
    same shape as `scripts/retrieve.py`'s CLI output — and attaches it as
    `QueryResponse.debug`. Off by default: it's 3 extra retrieval calls
    on top of the pipeline's own (multi-variant) retrieval, so it would
    roughly double per-query latency on this machine's BGE-M3/MPS setup
    if always on.
  - `GET /eval/summary` reads `data/eval/experiment_log.jsonl` (the same
    file `scripts/run_eval.py` appends to via `log_experiment`) on every
    request, not cached at startup — the dashboard should reflect a new
    `run_eval.py` run without an API restart.
- `src/api/models.py` defines `QueryRequest`/`QueryResponse` as a
  deliberately separate contract from `src/generation/models.py`'s
  `GeneratedAnswer`/`Citation` — an internal Phase 6 rename shouldn't
  silently break the frontend. `DebugStages`/`DebugHit` and
  `ExperimentRunOut`/`EvalSummaryResponse` follow the same pattern for
  the debug panel and eval dashboard.
  CORS is locked to `http://localhost:3000` (the Next.js dev origin).
- `frontend/`: Next.js 16 (App Router, TS, Tailwind v4) + shadcn/ui
  (`style: "base-nova"`, `@base-ui/react` primitives, not Radix — this is
  a pre-release Next.js/shadcn pairing, check `node_modules/next/dist/docs`
  before assuming Radix-era APIs). Two routes: `/` (chat, client
  component) and `/eval` (dashboard, server component — fetches
  `/eval/summary` at request time, no client JS needed for a read-only
  view).
  - `/`: textarea, submit on Enter, a "Retrieval debug" switch that sets
    `QueryRequest.debug`; renders answer + citation badges always, and
    a tabbed dense/sparse/hybrid/reranked hit table (shadcn `Tabs` +
    `Table`) only when `response.debug` came back non-null.
  - `/eval`: metric cards for the most recent logged run + a comparison
    table across every run in `experiment_log.jsonl`.
  - Visual style: bold/sticker-ish on the user's request ("as good
    looking as possible, keep behavior minimal") — thick black borders +
    hard offset shadows (`shadow-[Npx_Npx_0_0_#000]`, no blur) on
    `Card`/`Button`/`Textarea`, sticker-colored `Badge` variants
    (`amber`/`rose`/`violet`/`emerald` added alongside the default shadcn
    ones, used for citations/confidence), dotted background + two blurred
    color blobs in `layout.tsx` for the glow. This only touched
    `components/ui/*.tsx` base styles and `globals.css`/`layout.tsx` —
    no new behavior, same data flow as before the restyle.
- Verified end-to-end with Playwright (headless Chromium, installed
  ad hoc — not a repo dependency): asked "What was Apple's FY2023 net
  sales?" with `debug: true` against the real Qdrant index, got
  "$383,285 million [1], [3]" with confidence 1.00, two AAPL 10-K
  FY2023 Item 8 citations, and a populated 4-stage debug breakdown
  (dense top hit score 0.69, same Item 8 chunk). `/eval` dashboard
  screenshot-verified against the real `experiment_log.jsonl` (2 logged
  runs: `groq_smoke_test`, `smoke_test`).
  - Caution for future sessions: a stale `uvicorn` process from an
    earlier session was still holding the Qdrant embedded-mode lock
    (`data/index/qdrant_local` only allows one client) and serving
    pre-restyle code — had to `kill` it before the new code could bind
    the port. If `/query`/`/eval/summary` 500s with a Qdrant
    `AlreadyLocked` error, check for a leftover `uvicorn` process first.
  - Caution: BGE-M3 dense encoding on this machine's MPS device is
    inconsistent under concurrent load (single-variant batches ranged
    6s-50s when a second request hit the API at the same time) — don't
    fire concurrent test queries at a backend someone else (or another
    script) is actively using.
- Run it: `uvicorn src.api.main:app --port 8000` (backend) +
  `cd frontend && npm run dev` (frontend, port 3000).

## Experiment Log
| Experiment | Chunking | Retrieval | RAGAS Faithfulness | Hit Rate@5 |
|---|---|---|---|---|
| baseline | recursive-512 | dense-only | — | — |

## Phase 3 Notes
- Qdrant runs in embedded local mode: `QdrantClient(path=cfg.indexing.qdrant_path)`,
  storage at `data/index/qdrant_local/` — no Docker installed on this machine.
- One collection (`financial_rag`) holds both a named dense vector (`dense`,
  1024d, cosine, HNSW) and a named sparse vector (`sparse`, BM25) per point —
  Qdrant's native hybrid support, no second vector DB.
- `build_index.py` recreates the collection fresh each run (idempotent,
  not incremental) — corpus is still small and chunking strategy changes
  during ablation experiments, so a stale mix of old/new chunks would
  corrupt retrieval comparisons.
- BM25 vocab + idf persisted to `data/index/bm25_vocab.json` so Phase 5
  query-time sparse encoding stays consistent with what was indexed.
- Verified: 737/737 AAPL 10-K 2023 chunks indexed, both vector types
  queryable, dense search returns relevant hits.
