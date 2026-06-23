"""Unit tests for parse_llm_output (src/generation/output_parser.py)."""

from __future__ import annotations

from src.generation.models import Citation
from src.generation.output_parser import parse_llm_output


def _citation(citation_id: int) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        ticker="AAPL",
        form="10-K",
        fiscal_year=2023,
        item_label="Item 7",
    )


def test_parse_extracts_referenced_citations_in_order_of_first_appearance() -> None:
    available = [_citation(1), _citation(2), _citation(3)]
    raw = "Revenue grew [2] due to iPhone sales [1]. Margin also improved [2]."

    answer, citations, _ = parse_llm_output(raw, available)

    assert [c.citation_id for c in citations] == [2, 1]
    assert answer == raw


def test_parse_extracts_and_strips_trailing_confidence_line() -> None:
    available = [_citation(1)]
    raw = "Revenue was $383B [1].\nConfidence: 0.85"

    answer, citations, confidence = parse_llm_output(raw, available)

    assert answer == "Revenue was $383B [1]."
    assert confidence == 0.85
    assert citations[0].citation_id == 1


def test_parse_clamps_confidence_to_0_1_range() -> None:
    raw = "Answer text.\nConfidence: 1.5"
    _, _, confidence = parse_llm_output(raw, [])
    assert confidence == 1.0


def test_parse_ignores_citation_ids_not_in_available_list() -> None:
    available = [_citation(1)]
    raw = "Claim [1] and a hallucinated [99] reference."

    _, citations, _ = parse_llm_output(raw, available)

    assert [c.citation_id for c in citations] == [1]


def test_parse_with_no_citations_or_confidence() -> None:
    answer, citations, confidence = parse_llm_output("Not found in the provided filings.", [])
    assert answer == "Not found in the provided filings."
    assert citations == []
    assert confidence is None
