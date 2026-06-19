"""Attaches document metadata (ticker, year, section, page) to each chunk before indexing.

WHAT: enrich() stamps ChunkMetadata (filing identity + section identity) onto
every Chunk produced by chunkers.py.
WHY a separate step from chunking: chunk_*() functions in chunkers.py only
see raw section text — they have no filing identity. Keeping enrichment
separate means chunkers.py stays a pure text-in/chunks-out module, fully
testable without constructing a ParsedFiling.
"""

from __future__ import annotations

from src.ingestion.models import ParsedFiling, ParsedSection
from src.processing.models import Chunk, ChunkMetadata


def enrich(chunks: list[Chunk], filing: ParsedFiling, section: ParsedSection) -> list[Chunk]:
    """Mutate chunks in place, attaching filing + section identity. Returns the same list."""
    metadata = ChunkMetadata(
        ticker=filing.ticker,
        cik=filing.cik,
        form=filing.form,
        fiscal_year=filing.fiscal_year,
        company_name=filing.company_name,
        item_label=section.item_label,
        section_title=section.title,
    )
    for chunk in chunks:
        chunk.metadata = metadata
    return chunks
