"""Pydantic models for chunked filing text — output of Phase 2, input to Phase 3 indexing.

WHAT: Chunk is the unit that gets embedded + indexed. ChunkMetadata carries
the filing identity (ticker/cik/form/fiscal_year/section) so a retrieved
chunk is self-describing without a join back to ParsedFiling.
WHY pydantic: crosses the chunkers.py -> enricher.py -> Phase 3 indexing
boundary, per CLAUDE.md "use pydantic for data structures that cross module
boundaries".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.ingestion.models import FilingForm

ChunkStrategy = Literal["fixed", "recursive", "semantic", "sentence_window", "parent_child"]


class ChunkMetadata(BaseModel):
    """Filing identity stamped onto a chunk by enricher.py — makes a chunk self-describing."""

    ticker: str
    cik: str
    form: FilingForm
    fiscal_year: int
    company_name: str
    item_label: str
    section_title: str


class Chunk(BaseModel):
    """One unit of chunked text, ready for embedding once `metadata` is attached.

    WHY parent_id/is_parent instead of two separate models: parent-child
    chunking produces a mixed list (big context chunks + small indexed
    chunks pointing back at them) — one model with optional fields avoids a
    second near-duplicate class for what's structurally the same object.
    WHY context_text is separate from text: for sentence_window, `text` is
    the single sentence that gets embedded (precise match), `context_text`
    is the surrounding window handed to the LLM at generation time — these
    must stay different or embedding precision degrades.
    """

    chunk_id: str
    text: str
    strategy: ChunkStrategy
    chunk_index: int
    token_count: int
    parent_id: str | None = None
    is_parent: bool = False
    context_text: str | None = None
    metadata: ChunkMetadata | None = None
