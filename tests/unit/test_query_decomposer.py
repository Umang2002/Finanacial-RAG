"""Unit tests for multi-hop decomposition in src/query/query_decomposer.py.

WHY a fake LLM client instead of a real Ollama call: see test_query_transformer.py.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from src.query.query_decomposer import QueryDecomposer


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self.response


def _cfg(use_decomposition: bool):
    return OmegaConf.create({"query": {"use_decomposition": use_decomposition}})


def test_decompose_returns_raw_query_unchanged_when_disabled() -> None:
    llm = _FakeLLM("should not be used")
    decomposer = QueryDecomposer(_cfg(use_decomposition=False), llm=llm)

    result = decomposer.decompose("query")

    assert result.sub_questions == ["query"]
    assert llm.calls == []


def test_decompose_splits_into_sub_questions_when_enabled() -> None:
    llm = _FakeLLM("Which segment grew fastest in FY23?\nWhat drove that segment's growth?\n")
    decomposer = QueryDecomposer(_cfg(use_decomposition=True), llm=llm)

    result = decomposer.decompose("Which segment grew fastest and what drove that growth?")

    assert result.sub_questions == [
        "Which segment grew fastest in FY23?",
        "What drove that segment's growth?",
    ]


def test_decompose_falls_back_to_raw_query_on_empty_response() -> None:
    llm = _FakeLLM("   \n\n  ")
    decomposer = QueryDecomposer(_cfg(use_decomposition=True), llm=llm)

    result = decomposer.decompose("query")

    assert result.sub_questions == ["query"]
