"""Pydantic models for retrieval — output of Phase 5, input to Phase 6 generation.

WHAT: RetrievedChunk wraps one Chunk with the score(s) it was found by, so a
hit is self-describing about which retrieval stage(s) surfaced it.
WHY pydantic: crosses the retrieval/*.py -> generation/*.py module boundary,
per CLAUDE.md "use pydantic for data structures that cross module
boundaries".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.processing.models import Chunk

RetrievalSource = (
    str  # "dense" | "sparse" — kept as str so RRF can fuse N query variants per source
)


class RetrievedChunk(BaseModel):
    """One ranked hit: the chunk plus the score(s)/rank it earned at each retrieval stage.

    WHY score is a plain float, not per-source: dense cosine similarity and
    BM25 dot-product scores are on different, incomparable scales — only
    the rank within each source's list is meaningful, which is exactly what
    RRF fusion consumes. `score` here is whichever stage most recently set
    it (raw stage score for dense/sparse hits, fused RRF score after fusion,
    cross-encoder score after reranking) so callers always have "the
    current ranking score" without needing to know which stage produced it.
    """

    chunk: Chunk
    score: float
    sources: list[RetrievalSource] = Field(default_factory=list)
