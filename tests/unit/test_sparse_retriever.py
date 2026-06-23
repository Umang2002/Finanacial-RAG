"""Unit tests for SparseRetriever (src/retrieval/sparse_retriever.py).

WHY a real BM25SparseEncoder.fit() (not load_vocab from disk): exercises
the actual encode_query() math against a real in-memory Qdrant sparse
index, same spirit as test_sparse_indexer.py — load_vocab() itself is
already covered by test_sparse_indexer.py's persistence round-trip test.
"""

from __future__ import annotations

import uuid

from omegaconf import OmegaConf
from qdrant_client import QdrantClient, models

from src.indexing.dense_indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection
from src.indexing.sparse_indexer import BM25SparseEncoder
from src.retrieval.sparse_retriever import SparseRetriever


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "indexing": {
                "collection_name": "test_financial_rag",
                "distance_metric": "cosine",
                "hnsw_m": 16,
                "hnsw_ef_construct": 100,
                "bm25_vocab_path": "unused",
            },
            "retrieval": {"top_k_sparse": 5},
        }
    )


def _seeded_client_and_encoder(cfg) -> tuple[QdrantClient, BM25SparseEncoder]:
    texts = [
        "apple net sales grew in fiscal 2023",
        "microsoft cloud revenue increased in fiscal 2023",
        "apple stock price rose in late 2023",
        "tesla deliveries increased significantly",
    ]
    encoder = BM25SparseEncoder()
    encoder.fit(texts)
    sparse_vectors = encoder.encode_all()

    client = QdrantClient(location=":memory:")
    ensure_collection(
        client,
        cfg.indexing.collection_name,
        4,
        cfg.indexing.distance_metric,
        cfg.indexing.hnsw_m,
        cfg.indexing.hnsw_ef_construct,
    )
    point_ids = [str(uuid.uuid4()) for _ in texts]
    client.upsert(
        cfg.indexing.collection_name,
        points=[
            models.PointStruct(
                id=point_ids[i],
                vector={
                    DENSE_VECTOR_NAME: [0.0, 0.0, 0.0, 0.0],
                    SPARSE_VECTOR_NAME: sparse_vectors[i],
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
    return client, encoder


def test_search_returns_matching_chunk_by_keyword() -> None:
    cfg = _cfg()
    client, encoder = _seeded_client_and_encoder(cfg)
    retriever = SparseRetriever(cfg, client, encoder=encoder)

    hits = retriever.search("apple net sales")

    assert len(hits) >= 1
    assert hits[0].chunk.text.startswith("apple net sales")
    assert hits[0].sources == ["sparse"]


def test_search_returns_empty_for_out_of_vocab_query() -> None:
    cfg = _cfg()
    client, encoder = _seeded_client_and_encoder(cfg)
    retriever = SparseRetriever(cfg, client, encoder=encoder)

    hits = retriever.search("zzz nonexistent term")

    assert hits == []
