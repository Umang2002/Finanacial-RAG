"""Unit tests for RRF fusion logic and HybridRetriever (src/retrieval/hybrid_retriever.py)."""

from __future__ import annotations

import uuid

from omegaconf import OmegaConf
from qdrant_client import QdrantClient

from src.processing.models import Chunk
from src.retrieval import hybrid_retriever as hybrid_retriever_module
from src.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse
from src.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, text: str = "apple revenue") -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, strategy="recursive", chunk_index=0, token_count=2)


def _hit(chunk_id: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(chunk_id), score=score, sources=[source])


def test_rrf_fuse_sums_reciprocal_ranks_for_overlapping_chunk() -> None:
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    dense = [_hit(a, 0.9, "dense"), _hit(b, 0.8, "dense")]
    sparse = [_hit(a, 5.0, "sparse"), _hit(b, 3.0, "sparse")]

    fused = rrf_fuse([dense, sparse], k=60)

    by_id = {hit.chunk.chunk_id: hit for hit in fused}
    assert by_id[a].score == 2 * (1 / 61)
    assert by_id[b].score == 2 * (1 / 62)


def test_rrf_fuse_sorts_descending_by_fused_score() -> None:
    a, b, c = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    dense = [_hit(a, 0.9, "dense"), _hit(b, 0.8, "dense"), _hit(c, 0.7, "dense")]
    sparse = [_hit(c, 5.0, "sparse")]  # c also surfaced by sparse — should outrank a, b

    fused = rrf_fuse([dense, sparse])

    assert fused[0].chunk.chunk_id == c


def test_rrf_fuse_merges_sources_for_chunk_found_by_both() -> None:
    a = str(uuid.uuid4())
    dense = [_hit(a, 0.9, "dense")]
    sparse = [_hit(a, 5.0, "sparse")]

    fused = rrf_fuse([dense, sparse])

    assert fused[0].sources == ["dense", "sparse"]


def test_rrf_fuse_dedupes_chunk_appearing_once_per_list_into_one_hit() -> None:
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    dense = [_hit(a, 0.9, "dense"), _hit(b, 0.8, "dense")]

    fused = rrf_fuse([dense])

    assert len(fused) == 2


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "indexing": {"collection_name": "test_financial_rag"},
            "retrieval": {"top_k_dense": 10, "top_k_sparse": 10},
        }
    )


class _FakeDenseRetriever:
    def __init__(self, cfg, client) -> None:
        self.calls: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [_hit(f"dense-{query}", 1.0, "dense")]


class _FakeSparseRetriever:
    def __init__(self, cfg, client) -> None:
        self.calls: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [_hit(f"sparse-{query}", 1.0, "sparse")]


def test_hybrid_retriever_runs_dense_and_sparse_for_every_query_variant_and_fuses(
    monkeypatch,
) -> None:
    monkeypatch.setattr(hybrid_retriever_module, "DenseRetriever", _FakeDenseRetriever)
    monkeypatch.setattr(hybrid_retriever_module, "SparseRetriever", _FakeSparseRetriever)
    client = QdrantClient(location=":memory:")

    retriever = HybridRetriever(_cfg(), client)
    fused = retriever.retrieve(["raw question", "paraphrase"])

    fused_ids = {hit.chunk.chunk_id for hit in fused}
    assert fused_ids == {
        "dense-raw question",
        "dense-paraphrase",
        "sparse-raw question",
        "sparse-paraphrase",
    }
