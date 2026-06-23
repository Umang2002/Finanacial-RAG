"""Unit tests for HyDE and multi-query expansion in src/query/query_transformer.py.

WHY a fake LLM client instead of a real Ollama call: same reasoning as
faking SentenceTransformer in test_embeddings.py — no local server
dependency in unit tests, and we need to assert exactly what prompt/options
were sent.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from src.query.query_transformer import QueryTransformer


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self.response


def _cfg(use_hyde: bool = True, use_multi_query: bool = True, num_multi_queries: int = 3):
    return OmegaConf.create(
        {
            "query": {
                "use_hyde": use_hyde,
                "use_multi_query": use_multi_query,
                "num_multi_queries": num_multi_queries,
            }
        }
    )


def test_generate_hyde_returns_stripped_llm_output() -> None:
    llm = _FakeLLM("  Net sales increased 8% year over year to $383.3 billion.  ")
    transformer = QueryTransformer(_cfg(), llm=llm)

    result = transformer.generate_hyde("What was Apple's FY23 revenue?")

    assert result == "Net sales increased 8% year over year to $383.3 billion."
    assert "What was Apple's FY23 revenue?" in llm.calls[0]["prompt"]


def test_generate_multi_queries_filters_blank_and_duplicate_lines() -> None:
    llm = _FakeLLM(
        "What was Apple's FY23 revenue?\n"
        "What were Apple's net sales in fiscal 2023?\n"
        "\n"
        "How much revenue did Apple report for FY23?\n"
    )
    transformer = QueryTransformer(_cfg(), llm=llm)

    result = transformer.generate_multi_queries("What was Apple's FY23 revenue?")

    assert result == [
        "What were Apple's net sales in fiscal 2023?",
        "How much revenue did Apple report for FY23?",
    ]


def test_generate_multi_queries_caps_at_n() -> None:
    llm = _FakeLLM("a\nb\nc\nd\ne\n")
    transformer = QueryTransformer(_cfg(num_multi_queries=2), llm=llm)

    result = transformer.generate_multi_queries("query")

    assert result == ["a", "b"]


def test_transform_skips_hyde_when_disabled() -> None:
    llm = _FakeLLM("a\nb\nc\n")
    transformer = QueryTransformer(_cfg(use_hyde=False), llm=llm)

    result = transformer.transform("query")

    assert result.hyde_doc is None
    assert result.multi_queries == ["a", "b", "c"]


def test_transform_skips_multi_query_when_disabled() -> None:
    llm = _FakeLLM("hypothetical passage")
    transformer = QueryTransformer(_cfg(use_multi_query=False), llm=llm)

    result = transformer.transform("query")

    assert result.hyde_doc == "hypothetical passage"
    assert result.multi_queries == []


def test_transform_returns_raw_query() -> None:
    llm = _FakeLLM("x")
    transformer = QueryTransformer(_cfg(), llm=llm)

    result = transformer.transform("What was Apple's FY23 revenue?")

    assert result.raw_query == "What was Apple's FY23 revenue?"
