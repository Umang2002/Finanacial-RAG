"""Unit tests for dense_indexer.py (src/indexing/dense_indexer.py).

WHY QdrantClient(location=":memory:") instead of mocking: qdrant-client
ships a real in-process engine for this — exercising the actual collection
schema and upsert path costs nothing and catches API misuse that a mock
would hide.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.indexing.dense_indexer import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    upsert_points,
)


def _client() -> QdrantClient:
    return QdrantClient(location=":memory:")


def _ensure(client: QdrantClient, name: str, dim: int) -> None:
    ensure_collection(
        client, name, dense_dim=dim, distance_metric="cosine", hnsw_m=16, hnsw_ef_construct=100
    )


def test_ensure_collection_creates_dense_and_sparse_named_vectors() -> None:
    client = _client()
    _ensure(client, "test_col", dim=8)

    info = client.get_collection("test_col")
    assert info.config.params.vectors[DENSE_VECTOR_NAME].size == 8
    assert info.config.params.vectors[DENSE_VECTOR_NAME].distance == models.Distance.COSINE
    assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors


def test_ensure_collection_recreates_when_called_twice() -> None:
    client = _client()
    _ensure(client, "test_col", dim=4)
    point = models.PointStruct(
        id=1,
        vector={
            DENSE_VECTOR_NAME: [0.1, 0.2, 0.3, 0.4],
            SPARSE_VECTOR_NAME: models.SparseVector(indices=[0], values=[1.0]),
        },
        payload={},
    )
    upsert_points(client, "test_col", [point])
    assert client.get_collection("test_col").points_count == 1

    # recreate wipes the previous point
    _ensure(client, "test_col", dim=4)
    assert client.get_collection("test_col").points_count == 0


def test_upsert_points_batches_correctly() -> None:
    client = _client()
    _ensure(client, "test_col", dim=2)

    points = [
        models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME: [0.1 * i, 0.2 * i],
                SPARSE_VECTOR_NAME: models.SparseVector(indices=[0], values=[1.0]),
            },
            payload={"i": i},
        )
        for i in range(5)
    ]

    upsert_points(client, "test_col", points, batch_size=2)  # forces 3 batches

    assert client.get_collection("test_col").points_count == 5
