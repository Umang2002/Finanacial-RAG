# Financial RAG — Project Context for Claude Code

## What This Project Is
End-to-end Hybrid Search RAG system for SEC financial document intelligence.
Built for learning (understand every component) and resume (production-grade patterns).
**100% free stack — no API costs.**

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
- **Ollama**: free local LLM server; runs llama3.2, mistral, phi3 with no API cost
- **RRF (Reciprocal Rank Fusion)**: simple, parameter-free fusion — outperforms weighted sum in practice
- **RAGAS**: LLM-as-judge eval framework — measures RAG quality without human labels
- **FinanceBench**: open QA benchmark over real SEC filings — gives credible, comparable eval numbers

## Free Stack — Zero API Costs
| Component | Tool | Cost |
|---|---|---|
| Embeddings | BGE-M3 (sentence-transformers) | Free, local |
| LLM | Ollama + llama3.2 | Free, local |
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
# 1. Install Ollama → https://ollama.com/download
ollama pull llama3.2:3b     # ~2GB download — tag must match exactly (no bare `llama3.2` alias)
# 2. BGE-M3 downloads automatically on first use (~2.3GB)
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
Phase 5 — Retrieval (implemented, smoke-testing)

## Planned: Phase 8 — Frontend + Deployment (after Phase 6+7 done, NOT now)
- Next.js frontend, deployed (resume-facing, public link).
- Backend API layer: FastAPI wrapping RetrievalPipeline + generation (`/query`, `/health`). Chosen over Next.js-routes-calling-Python.
- Frontend v1 scope: chat-style query input + streamed answer w/ citations, retrieval debug panel (dense/sparse/hybrid/reranked per-stage view, like `scripts/retrieve.py`), eval dashboard (RAGAS + retrieval metrics).
- Deployment: swap Ollama → Groq free-tier API for the deployed backend (Ollama needs persistent compute, doesn't fit free serverless hosts) — local dev keeps Ollama. This breaks "100% free local stack" for prod only; local stack unchanged.
- Do not start this until Phases 6 (Generation) and 7 (Evaluation) are done.

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
