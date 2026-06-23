"""Unit tests for RetrievalPipeline / build_query_variants (src/retrieval/retrieval_pipeline.py)."""

from __future__ import annotations

from omegaconf import OmegaConf
from qdrant_client import QdrantClient

from src.processing.models import Chunk
from src.query.models import DecomposedQuery, TransformedQuery
from src.retrieval import retrieval_pipeline as retrieval_pipeline_module
from src.retrieval.models import RetrievedChunk
from src.retrieval.retrieval_pipeline import RetrievalPipeline, build_query_variants


def test_build_query_variants_includes_raw_query_first() -> None:
    variants = build_query_variants("what was revenue?")
    assert variants == ["what was revenue?"]


def test_build_query_variants_adds_hyde_and_multi_queries() -> None:
    transformed = TransformedQuery(
        raw_query="q",
        hyde_doc="hypothetical passage",
        multi_queries=["paraphrase a", "paraphrase b"],
    )
    variants = build_query_variants("q", transformed=transformed)
    assert variants == ["q", "hypothetical passage", "paraphrase a", "paraphrase b"]


def test_build_query_variants_adds_sub_questions() -> None:
    decomposed = DecomposedQuery(raw_query="q", sub_questions=["sub a", "sub b"])
    variants = build_query_variants("q", decomposed=decomposed)
    assert variants == ["q", "sub a", "sub b"]


def test_build_query_variants_dedupes_raw_query_repeated_by_decomposition() -> None:
    # query_decomposer.py returns [raw_query] unchanged when decomposition is off/skipped
    decomposed = DecomposedQuery(raw_query="q", sub_questions=["q"])
    variants = build_query_variants("q", decomposed=decomposed)
    assert variants == ["q"]


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {"indexing": {"collection_name": "test"}, "retrieval": {"reranker": "bge"}}
    )


def _hit(text: str) -> RetrievedChunk:
    chunk = Chunk(chunk_id=text, text=text, strategy="recursive", chunk_index=0, token_count=1)
    return RetrievedChunk(chunk=chunk, score=1.0, sources=["dense"])


class _FakeHybridRetriever:
    def __init__(self, cfg, client) -> None:
        self.queries_seen: list[str] = []

    def retrieve(self, queries: list[str]) -> list[RetrievedChunk]:
        self.queries_seen = queries
        return [_hit("fused chunk")]


class _FakeReranker:
    def __init__(self, cfg, model=None) -> None:
        self.query_seen: str | None = None

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k=None
    ) -> list[RetrievedChunk]:
        self.query_seen = query
        return candidates


def test_retrieve_wires_variants_through_hybrid_and_reranker(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_pipeline_module, "HybridRetriever", _FakeHybridRetriever)
    monkeypatch.setattr(retrieval_pipeline_module, "Reranker", _FakeReranker)
    client = QdrantClient(location=":memory:")

    pipeline = RetrievalPipeline(_cfg(), client=client)
    transformed = TransformedQuery(raw_query="q", hyde_doc="doc", multi_queries=["p1"])
    results = pipeline.retrieve("q", transformed=transformed)

    assert pipeline.hybrid.queries_seen == ["q", "doc", "p1"]
    assert pipeline.reranker.query_seen == "q"
    assert results[0].chunk.text == "fused chunk"
