"""Unit tests for Generator (src/generation/generator.py).

WHY a fake OllamaClient: the test only needs to verify generate() wires
context assembly -> prompt -> LLM call -> output parsing correctly, not
exercise a real local model.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.generation.generator import Generator
from src.processing.models import Chunk, ChunkMetadata
from src.retrieval.models import RetrievedChunk


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return self.response


def _cfg(citation_style: str = "inline") -> OmegaConf:
    return OmegaConf.create({"generation": {"citation_style": citation_style, "temperature": 0.0}})


def _hit(text: str) -> RetrievedChunk:
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
    return RetrievedChunk(chunk=chunk, score=0.0, sources=["dense"])


def test_generate_returns_grounded_answer_with_resolved_citations() -> None:
    fake_llm = _FakeLLM("Net sales were $383B [1].\nConfidence: 0.9")
    generator = Generator(_cfg(), llm=fake_llm)

    result = generator.generate("What were net sales?", [_hit("Net sales were $383.3 billion.")])

    assert result.query == "What were net sales?"
    assert result.answer == "Net sales were $383B [1]."
    assert result.confidence == 0.9
    assert len(result.citations) == 1
    assert result.citations[0].ticker == "AAPL"


def test_generate_passes_assembled_context_and_query_into_prompt() -> None:
    fake_llm = _FakeLLM("answer")
    generator = Generator(_cfg(), llm=fake_llm)

    generator.generate("my question", [_hit("relevant passage")])

    assert "relevant passage" in fake_llm.last_prompt
    assert "my question" in fake_llm.last_prompt
    assert fake_llm.last_system is not None


def test_generate_handles_no_retrieved_chunks() -> None:
    fake_llm = _FakeLLM("Not found in the provided filings.")
    generator = Generator(_cfg(), llm=fake_llm)

    result = generator.generate("unanswerable question", [])

    assert result.citations == []
    assert result.answer == "Not found in the provided filings."


def test_generator_rejects_unsupported_citation_style() -> None:
    with pytest.raises(ValueError, match="citation_style"):
        Generator(_cfg(citation_style="footnote"), llm=_FakeLLM(""))
