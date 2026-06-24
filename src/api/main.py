"""FastAPI app: wraps the Phase 4-6 query pipeline behind /health and /query.

WHAT: on startup, builds every pipeline component once (QueryAnalyzer,
QueryTransformer, QueryDecomposer, RetrievalPipeline, Generator) and stashes
them on app.state. POST /query then just runs the same
intent -> transform -> decompose(if multi-hop) -> retrieve -> generate
sequence scripts/generate.py uses, and returns it as JSON.
WHY build once at startup, not per-request: RetrievalPipeline/Generator load
BGE-M3 + BGE-reranker (local sentence-transformers models) — see Phase 6/7
notes in CLAUDE.md, that load alone is most of scripts/generate.py's ~1.5min
runtime. Rebuilding per request would make every query pay that cost.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import CitationOut, QueryRequest, QueryResponse
from src.generation.generator import Generator
from src.query.query_analyzer import QueryAnalyzer
from src.query.query_decomposer import QueryDecomposer
from src.query.query_transformer import QueryTransformer
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config + every pipeline component once, before the first request is served."""
    cfg = load_config()
    app.state.analyzer = QueryAnalyzer(cfg)
    app.state.transformer = QueryTransformer(cfg)
    app.state.decomposer = QueryDecomposer(cfg)
    app.state.retrieval_pipeline = RetrievalPipeline(cfg)
    app.state.generator = Generator(cfg)
    logger.info("Pipeline components loaded — API ready")
    yield


app = FastAPI(title="Financial RAG API", lifespan=lifespan)

# LEARN: Next.js dev server runs on :3000, this API on :8000 — different
# origins, so the browser blocks the fetch without explicit CORS allowance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — confirms the process is up, not that models are loaded."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full Phase 4-6 pipeline for one question and return the cited answer."""
    analyzed = app.state.analyzer.analyze(request.query)
    transformed = app.state.transformer.transform(request.query)
    decomposed = app.state.decomposer.decompose(request.query) if analyzed.is_multi_hop else None

    chunks = app.state.retrieval_pipeline.retrieve(
        request.query, transformed=transformed, decomposed=decomposed
    )
    result = app.state.generator.generate(request.query, chunks)

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        citations=[
            CitationOut(
                citation_id=c.citation_id,
                ticker=c.ticker,
                form=c.form,
                fiscal_year=c.fiscal_year,
                item_label=c.item_label,
            )
            for c in result.citations
        ],
    )
