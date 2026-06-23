"""Unit tests for intent classification in src/query/query_analyzer.py.

WHY a fake LLM client instead of a real Ollama call: see test_query_transformer.py.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from src.query.query_analyzer import QueryAnalyzer


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self.response


def _cfg():
    return OmegaConf.create({})


def test_analyze_parses_exact_label() -> None:
    analyzer = QueryAnalyzer(_cfg(), llm=_FakeLLM("comparison"))

    result = analyzer.analyze("How did Apple's margin compare to Microsoft's?")

    assert result.intent == "comparison"
    assert result.is_multi_hop is False


def test_analyze_parses_label_embedded_in_sentence() -> None:
    analyzer = QueryAnalyzer(_cfg(), llm=_FakeLLM("The label is: multi_hop."))

    result = analyzer.analyze("Which segment grew fastest and what drove that growth?")

    assert result.intent == "multi_hop"
    assert result.is_multi_hop is True


def test_analyze_defaults_to_other_on_unknown_response() -> None:
    analyzer = QueryAnalyzer(_cfg(), llm=_FakeLLM("not a real label"))

    result = analyzer.analyze("query")

    assert result.intent == "other"
    assert result.is_multi_hop is False


def test_analyze_returns_raw_query() -> None:
    analyzer = QueryAnalyzer(_cfg(), llm=_FakeLLM("factual_lookup"))

    result = analyzer.analyze("What was Apple's FY23 revenue?")

    assert result.raw_query == "What was Apple's FY23 revenue?"
