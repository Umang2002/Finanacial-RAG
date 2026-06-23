"""Pydantic models for generation — output of Phase 6, input to Phase 7 evaluation.

WHAT: Citation ties one bracket-numbered reference in the LLM's answer back
to the filing it came from. GeneratedAnswer is the final structured result:
answer text, the citations actually used, and an optional self-reported
confidence score.
WHY pydantic: crosses the generation/*.py -> evaluation/*.py module
boundary, per CLAUDE.md "use pydantic for data structures that cross module
boundaries".
"""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """One [n] reference in an answer, resolved back to its source chunk."""

    citation_id: int
    chunk_id: str
    ticker: str
    form: str
    fiscal_year: int
    item_label: str


class GeneratedAnswer(BaseModel):
    """Final Phase 6 output: grounded answer text plus the citations it relied on."""

    query: str
    answer: str
    citations: list[Citation]
    confidence: float | None = None


class AssembledContext(BaseModel):
    """ContextAssembler's output: the prompt-ready context block plus every citation it defines.

    WHY citations carries every chunk handed to the LLM (not just the ones
    it ends up citing): output_parser only knows which [n] ids appear in
    the raw answer text — it needs this full list to resolve them back to
    filing identity.
    """

    text: str
    citations: list[Citation]
