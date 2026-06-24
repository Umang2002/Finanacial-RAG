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

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    CitationOut,
    DebugHit,
    DebugStages,
    EvalSummaryResponse,
    ExperimentRunOut,
    QueryRequest,
    QueryResponse,
    RagasMetricsOut,
    RetrievalMetricsOut,
)
from src.generation.generator import Generator
from src.query.query_analyzer import QueryAnalyzer
from src.query.query_decomposer import QueryDecomposer
from src.query.query_transformer import QueryTransformer
from src.retrieval.hybrid_retriever import rrf_fuse
from src.retrieval.models import RetrievedChunk
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config + every pipeline component once, before the first request is served."""
    cfg = load_config()
    app.state.eval_log_cfg = cfg
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
# CORS_ORIGINS (comma-separated) lets the deployed Vercel origin in without
# hardcoding it — localhost:3000 stays allowed for local dev either way.
_extra_origins = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", *_extra_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


_DEBUG_TOP_K = 5


def _to_debug_hits(hits: list[RetrievedChunk]) -> list[DebugHit]:
    """Reshape the top _DEBUG_TOP_K RetrievedChunk hits into the frontend's debug-row shape."""
    rows: list[DebugHit] = []
    for hit in hits[:_DEBUG_TOP_K]:
        meta = hit.chunk.metadata
        rows.append(
            DebugHit(
                score=hit.score,
                ticker=meta.ticker if meta else "—",
                form=meta.form if meta else "—",
                fiscal_year=meta.fiscal_year if meta else 0,
                item_label=meta.item_label if meta else "—",
                text_preview=hit.chunk.text[:160].replace("\n", " "),
            )
        )
    return rows


def _run_debug_stages(raw_query: str) -> DebugStages:
    """Re-run dense/sparse/hybrid/reranked stages standalone, same flow as scripts/retrieve.py.

    WHY a second, simpler run instead of instrumenting RetrievalPipeline.retrieve():
    that method already fuses every query variant (HyDE/multi-query/decomposition)
    into one pass for generation quality — splitting it mid-flight to capture
    per-stage snapshots would mean threading debug-only state through Phase 5
    internals. This mirrors the CLI's debug view instead: dense/sparse/hybrid/
    reranked over the raw query alone, which is what a human inspects a query's
    retrieval behavior with.
    """
    pipeline = app.state.retrieval_pipeline
    dense_hits = pipeline.hybrid.dense.search(raw_query)
    sparse_hits = pipeline.hybrid.sparse.search(raw_query)
    fused_hits = rrf_fuse([dense_hits, sparse_hits])
    reranked_hits = pipeline.reranker.rerank(raw_query, fused_hits)

    return DebugStages(
        dense=_to_debug_hits(dense_hits),
        sparse=_to_debug_hits(sparse_hits),
        hybrid=_to_debug_hits(fused_hits),
        reranked=_to_debug_hits(reranked_hits),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — confirms the process is up, not that models are loaded."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full Phase 4-6 pipeline for one question and return the cited answer.

    If request.debug is set, also runs the standalone dense/sparse/hybrid/
    reranked stage breakdown (see _run_debug_stages) for the retrieval debug panel.
    """
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
        debug=_run_debug_stages(request.query) if request.debug else None,
    )


@app.get("/eval/summary", response_model=EvalSummaryResponse)
def eval_summary() -> EvalSummaryResponse:
    """Read every logged experiment run for the eval dashboard, most recent first.

    WHY read the JSONL on every request rather than caching at startup: this
    file only grows when someone runs scripts/run_eval.py, which isn't part
    of the API process — caching at lifespan-build time would show a stale
    dashboard until the next API restart.
    """
    cfg = app.state.eval_log_cfg
    log_path = Path(cfg.evaluation.dataset).parent / "experiment_log.jsonl"
    if not log_path.exists():
        return EvalSummaryResponse(runs=[])

    runs: list[ExperimentRunOut] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row["metrics"]
        generation_cfg = row.get("config", {}).get("generation", {})
        runs.append(
            ExperimentRunOut(
                config_name=row["config_name"],
                num_examples=row["num_examples"],
                generation_provider=generation_cfg.get("provider"),
                generation_model=generation_cfg.get("model"),
                retrieval=RetrievalMetricsOut(
                    hit_rate=metrics["hit_rate"],
                    mrr=metrics["mrr"],
                    ndcg=metrics["ndcg"],
                    precision_at_k=metrics["precision_at_k"],
                    recall_at_k=metrics["recall_at_k"],
                ),
                ragas=RagasMetricsOut(
                    faithfulness=metrics["faithfulness"],
                    answer_relevancy=metrics["answer_relevancy"],
                    context_precision=metrics["context_precision"],
                    context_recall=metrics["context_recall"],
                ),
            )
        )
    runs.reverse()
    return EvalSummaryResponse(runs=runs)
