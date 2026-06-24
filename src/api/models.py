"""Pydantic request/response models for the Phase 8 API — the frontend's only contract.

WHY a separate module from src/generation/models.py: Citation/GeneratedAnswer
are the internal Phase 6 result shape; QueryResponse is the over-the-wire
shape the Next.js frontend depends on. Keeping them distinct means an
internal refactor (e.g. renaming GeneratedAnswer fields) can't silently
break the API contract.
"""

from __future__ import annotations

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Body of POST /query — the question, plus an opt-in flag for retrieval debug stages."""

    query: str
    # LEARN: debug stages cost 3 extra retrieval calls (dense/sparse/fused
    # run again outside the pipeline to capture each stage) — default False
    # so the common case stays as fast as the original /query.
    debug: bool = False


class CitationOut(BaseModel):
    """One [n] citation in the answer, resolved to its source filing."""

    citation_id: int
    ticker: str
    form: str
    fiscal_year: int
    item_label: str


class DebugHit(BaseModel):
    """One chunk's result at a single retrieval stage — score + enough identity to render a row."""

    score: float
    ticker: str
    form: str
    fiscal_year: int
    item_label: str
    text_preview: str


class DebugStages(BaseModel):
    """Dense/sparse/hybrid/reranked top hits for one query — mirrors scripts/retrieve.py."""

    dense: list[DebugHit]
    sparse: list[DebugHit]
    hybrid: list[DebugHit]
    reranked: list[DebugHit]


class QueryResponse(BaseModel):
    """Body of the POST /query response — answer + citations the frontend renders."""

    query: str
    answer: str
    confidence: float | None
    citations: list[CitationOut]
    debug: DebugStages | None = None


class RetrievalMetricsOut(BaseModel):
    """Mean retrieval metrics for one experiment run."""

    hit_rate: float
    mrr: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float


class RagasMetricsOut(BaseModel):
    """Mean RAGAS metrics for one experiment run — any may be null if that metric errored."""

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


class ExperimentRunOut(BaseModel):
    """One row of data/eval/experiment_log.jsonl, reshaped for the eval dashboard."""

    config_name: str
    num_examples: int
    generation_provider: str | None
    generation_model: str | None
    retrieval: RetrievalMetricsOut
    ragas: RagasMetricsOut


class EvalSummaryResponse(BaseModel):
    """Body of GET /eval/summary — every logged experiment run, most recent first."""

    runs: list[ExperimentRunOut]
