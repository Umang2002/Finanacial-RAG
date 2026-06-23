"""Unit tests for ContextAssembler (src/generation/context_assembler.py)."""

from __future__ import annotations

from omegaconf import OmegaConf

from src.generation.context_assembler import ContextAssembler, _sandwich_reorder
from src.processing.models import Chunk, ChunkMetadata
from src.retrieval.models import RetrievedChunk


def _hit(text: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=text,
        text=text,
        strategy="recursive",
        chunk_index=0,
        token_count=1,
        metadata=ChunkMetadata(
            ticker="AAPL",
            cik="320193",
            form="10-K",
            fiscal_year=2023,
            company_name="Apple Inc.",
            item_label="Item 7",
            section_title="MD&A",
        ),
    )
    return RetrievedChunk(chunk=chunk, score=score, sources=["dense"])


def _cfg() -> OmegaConf:
    return OmegaConf.create({})


def test_sandwich_reorder_places_most_relevant_at_both_ends() -> None:
    chunks = [_hit("1"), _hit("2"), _hit("3"), _hit("4"), _hit("5")]
    reordered = _sandwich_reorder(chunks)
    assert [c.chunk.text for c in reordered] == ["1", "3", "5", "4", "2"]


def test_assemble_assigns_citation_ids_in_relevance_order() -> None:
    chunks = [_hit("best"), _hit("middle"), _hit("worst")]
    assembled = ContextAssembler(_cfg()).assemble(chunks)

    assert [c.citation_id for c in assembled.citations] == [1, 2, 3]
    assert assembled.citations[0].chunk_id == "best"
    assert assembled.citations[0].ticker == "AAPL"
    assert assembled.citations[0].fiscal_year == 2023


def test_assemble_text_contains_bracketed_citation_for_each_chunk() -> None:
    chunks = [_hit("best"), _hit("worst")]
    assembled = ContextAssembler(_cfg()).assemble(chunks)

    assert "[1]" in assembled.text
    assert "[2]" in assembled.text
    assert "best" in assembled.text
    assert "worst" in assembled.text


def test_assemble_handles_empty_chunks() -> None:
    assembled = ContextAssembler(_cfg()).assemble([])
    assert assembled.text == ""
    assert assembled.citations == []
