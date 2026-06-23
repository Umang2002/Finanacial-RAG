"""Unit tests for IndexManager (src/indexing/index_manager.py).

WHY monkeypatch QdrantClient + BGEEmbedder: this is the orchestrator —
the test exercises the real wiring (ensure_collection -> embed -> BM25 fit
-> upsert) without downloading BGE-M3 or touching disk for Qdrant storage.
QdrantClient is redirected to in-memory mode; BGEEmbedder is swapped for a
fake returning fixed-size vectors matching the test's small dense_dimensions.
"""

from __future__ import annotations

import uuid

import numpy as np
from omegaconf import OmegaConf
from qdrant_client import QdrantClient

from src.indexing import index_manager as index_manager_module
from src.indexing.index_manager import IndexManager
from src.processing.models import Chunk


class _FakeEmbedder:
    def __init__(self, model_name: str, batch_size: int) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=float)


def _cfg(tmp_path):
    return OmegaConf.create(
        {
            "embeddings": {"dense_model": "fake-model", "dense_dimensions": 4, "batch_size": 8},
            "indexing": {
                "collection_name": "test_financial_rag",
                "distance_metric": "cosine",
                "hnsw_m": 16,
                "hnsw_ef_construct": 100,
                "qdrant_path": str(tmp_path / "qdrant_local"),
                "bm25_vocab_path": str(tmp_path / "bm25_vocab.json"),
            },
        }
    )


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text=f"apple revenue grew in fiscal year {i}",
            strategy="recursive",
            chunk_index=i,
            token_count=7,
        )
        for i in range(n)
    ]


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(index_manager_module, "BGEEmbedder", _FakeEmbedder)
    monkeypatch.setattr(
        index_manager_module,
        "QdrantClient",
        lambda path=None, **kwargs: QdrantClient(location=":memory:"),
    )


def test_build_index_returns_chunk_count(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch)
    manager = IndexManager(_cfg(tmp_path))

    n_points = manager.build_index(_make_chunks(5))

    assert n_points == 5


def test_build_index_upserts_into_qdrant_with_both_vectors(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch)
    cfg = _cfg(tmp_path)
    manager = IndexManager(cfg)

    manager.build_index(_make_chunks(3))

    info = manager.client.get_collection(cfg.indexing.collection_name)
    assert info.points_count == 3

    point = manager.client.scroll(cfg.indexing.collection_name, limit=1, with_vectors=True)[0][0]
    assert len(point.vector["dense"]) == 4
    assert len(point.vector["sparse"].indices) > 0
    assert point.payload["text"].startswith("apple revenue")


def test_build_index_persists_bm25_vocab(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch)
    cfg = _cfg(tmp_path)
    manager = IndexManager(cfg)

    manager.build_index(_make_chunks(3))

    assert (tmp_path / "bm25_vocab.json").exists()
