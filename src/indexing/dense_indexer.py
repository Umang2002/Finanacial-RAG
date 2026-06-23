"""Qdrant collection setup: one collection holding both a dense (BGE-M3) and a
sparse (BM25) named vector per point — Qdrant's hybrid support means no
second vector database is needed (see CLAUDE.md "Key Tech Choices").
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.utils.logging import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    distance_metric: str,
    hnsw_m: int,
    hnsw_ef_construct: int,
) -> None:
    """Recreate the collection fresh with both a dense and a sparse named vector.

    WHY recreate (delete-if-exists then create) rather than incremental
    upsert: the corpus is still small and changing rapidly during chunking
    experiments (configs/experiments/chunking_ablation.yaml) — a stale mix
    of old+new chunking strategies in one collection would silently corrupt
    retrieval comparisons. Switch to incremental upsert once the dataset is
    large enough that a full rebuild is expensive.
    """
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists — recreating")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_dim,
                distance=_DISTANCE_MAP[distance_metric],
                hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )
    logger.info(
        f"Created collection '{collection_name}' "
        f"(dense={dense_dim}d/{distance_metric}, sparse=bm25)"
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int = 100,
) -> None:
    """Upsert points in batches so one giant request doesn't block on embedded local-mode I/O.

    Args:
        client: connected QdrantClient (embedded local-mode or server).
        collection_name: target collection, already created via ensure_collection.
        points: PointStructs with both named vectors + chunk payload set.
        batch_size: points per upsert call.
    """
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
    logger.info(f"Upserted {len(points)} points into '{collection_name}'")
