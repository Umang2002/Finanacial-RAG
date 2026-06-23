"""Unit tests for DenseRetriever (src/retrieval/dense_retriever.py).

WHY a real in-memory QdrantClient, fake embedder: same pattern as
test_index_manager.py — exercises the real query_points() call against
Qdrant's :memory: mode without downloading BGE-M3.
"""

from __future__ import annotations

import uuid

import numpy as np
from omegaconf import OmegaConf
from qdrant_client import QdrantClient, models

from src.indexing.dense_indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection
from src.retrieval.dense_retriever import DenseRetriever


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts])


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "embeddings": {"dense_model": "fake", "dense_dimensions": 4, "batch_size": 8},
            "indexing": {
                "collection_name": "test_financial_rag",
                "distance_metric": "cosine",
                "hnsw_m": 16,
                "hnsw_ef_construct": 100,
            },
            "retrieval": {"top_k_dense": 5},
        }
    )


def _seeded_client(cfg) -> QdrantClient:
    client = QdrantClient(location=":memory:")
    ensure_collection(
        client,
        cfg.indexing.collection_name,
        cfg.embeddings.dense_dimensions,
        cfg.indexing.distance_metric,
        cfg.indexing.hnsw_m,
        cfg.indexing.hnsw_ef_construct,
    )
    texts = ["apple net sales grew in fiscal 2023", "microsoft cloud revenue increased"]
    point_ids = [str(uuid.uuid4()) for _ in texts]
    client.upsert(
        cfg.indexing.collection_name,
        points=[
            models.PointStruct(
                id=point_ids[i],
                vector={
                    DENSE_VECTOR_NAME: [1.0, 0.0, 0.0, 0.0],
                    SPARSE_VECTOR_NAME: models.SparseVector(indices=[], values=[]),
                },
                payload={
                    "chunk_id": point_ids[i],
                    "text": texts[i],
                    "strategy": "recursive",
                    "chunk_index": i,
                    "token_count": 7,
                },
            )
            for i in range(len(texts))
        ],
        wait=True,
    )
    return client


def test_search_returns_chunk_with_score() -> None:
    cfg = _cfg()
    client = _seeded_client(cfg)
    retriever = DenseRetriever(cfg, client, embedder=_FakeEmbedder())

    hits = retriever.search("what were apple's net sales?")

    assert len(hits) == 2
    assert hits[0].sources == ["dense"]
    assert hits[0].score > 0


def test_search_respects_top_k_override() -> None:
    cfg = _cfg()
    client = _seeded_client(cfg)
    retriever = DenseRetriever(cfg, client, embedder=_FakeEmbedder())

    hits = retriever.search("query", top_k=1)

    assert len(hits) == 1
