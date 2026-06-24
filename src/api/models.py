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
    """Body of POST /query — just the raw question."""

    query: str


class CitationOut(BaseModel):
    """One [n] citation in the answer, resolved to its source filing."""

    citation_id: int
    ticker: str
    form: str
    fiscal_year: int
    item_label: str


class QueryResponse(BaseModel):
    """Body of the POST /query response — answer + citations the frontend renders."""

    query: str
    answer: str
    confidence: float | None
    citations: list[CitationOut]
