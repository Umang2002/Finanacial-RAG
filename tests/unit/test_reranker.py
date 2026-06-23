"""Unit tests for Reranker (src/retrieval/reranker.py).

WHY a fake CrossEncoder: loading BAAI/bge-reranker-v2-m3 is a real model
download — the test only needs to verify rerank() wires predict() output
into a re-sorted, truncated RetrievedChunk list.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from src.processing.models import Chunk
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import Reranker


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs_seen: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs_seen = pairs
        return self.scores


def _cfg() -> OmegaConf:
    return OmegaConf.create({"retrieval": {"reranker": "bge", "top_k_rerank": 2}})


def _hit(text: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(chunk_id=text, text=text, strategy="recursive", chunk_index=0, token_count=1)
    return RetrievedChunk(chunk=chunk, score=score, sources=["dense"])


def test_rerank_sorts_by_cross_encoder_score_and_truncates_to_top_k() -> None:
    candidates = [_hit("low relevance"), _hit("high relevance"), _hit("mid relevance")]
    fake_model = _FakeCrossEncoder(scores=[0.1, 0.9, 0.5])
    reranker = Reranker(_cfg(), model=fake_model)

    results = reranker.rerank("query", candidates)

    assert [hit.chunk.text for hit in results] == ["high relevance", "mid relevance"]
    assert results[0].score == 0.9


def test_rerank_passes_original_query_paired_with_each_candidate_text() -> None:
    candidates = [_hit("chunk a"), _hit("chunk b")]
    fake_model = _FakeCrossEncoder(scores=[0.0, 0.0])
    reranker = Reranker(_cfg(), model=fake_model)

    reranker.rerank("original question", candidates)

    assert fake_model.pairs_seen == [
        ("original question", "chunk a"),
        ("original question", "chunk b"),
    ]


def test_rerank_handles_empty_candidates() -> None:
    fake_model = _FakeCrossEncoder(scores=[])
    reranker = Reranker(_cfg(), model=fake_model)

    assert reranker.rerank("query", []) == []
