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
| Vector DB | Qdrant (Docker) | Free, local |
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
# 1. Install Docker Desktop → https://www.docker.com/products/docker-desktop/
# 2. Install Ollama → https://ollama.com/download
ollama pull llama3.2        # ~2GB download
# 3. BGE-M3 downloads automatically on first use (~2.3GB)
```

## Running the Project
```bash
# Start Qdrant
docker compose up -d

# Install deps
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Download filings
python scripts/download_filings.py --ticker AAPL --years 2022 2023

# Build index (runs full ingestion → processing → indexing)
python scripts/build_index.py --config configs/base.yaml

# Run evaluation
python scripts/run_eval.py --config configs/base.yaml
```

## Current Phase
Phase 1 — Data Ingestion (in progress)

## Experiment Log
| Experiment | Chunking | Retrieval | RAGAS Faithfulness | Hit Rate@5 |
|---|---|---|---|---|
| baseline | recursive-512 | dense-only | — | — |
