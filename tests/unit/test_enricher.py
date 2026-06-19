"""Unit tests for enrich() (src/processing/enricher.py)."""

from __future__ import annotations

from datetime import date

from src.ingestion.models import ParsedFiling, ParsedSection
from src.processing.chunkers import chunk_fixed
from src.processing.enricher import enrich


def _make_filing() -> ParsedFiling:
    return ParsedFiling(
        ticker="AAPL",
        cik="0000320193",
        form="10-K",
        fiscal_year=2023,
        company_name="Apple Inc.",
        period_end_date=date(2023, 9, 30),
        full_text="irrelevant",
        sections=[],
        raw_tables=[],
    )


def test_enrich_stamps_filing_and_section_identity_onto_every_chunk() -> None:
    filing = _make_filing()
    section = ParsedSection(
        item_label="Item 7", title="MD&A", text="Revenue grew 10% year over year."
    )
    chunks = chunk_fixed(section.text, chunk_size=20, chunk_overlap=0)

    enriched = enrich(chunks, filing, section)

    assert len(enriched) == len(chunks)
    for chunk in enriched:
        assert chunk.metadata is not None
        assert chunk.metadata.ticker == "AAPL"
        assert chunk.metadata.cik == "0000320193"
        assert chunk.metadata.form == "10-K"
        assert chunk.metadata.fiscal_year == 2023
        assert chunk.metadata.company_name == "Apple Inc."
        assert chunk.metadata.item_label == "Item 7"
        assert chunk.metadata.section_title == "MD&A"


def test_enrich_returns_same_list_object_mutated_in_place() -> None:
    filing = _make_filing()
    section = ParsedSection(item_label="Item 1", title="Business", text="Some text.")
    chunks = chunk_fixed(section.text, chunk_size=20, chunk_overlap=0)

    result = enrich(chunks, filing, section)
    assert result is chunks
