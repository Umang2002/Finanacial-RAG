"""Orchestrates dense + sparse indexing; single entry point for Phase 3.

WHAT: IndexManager.build_index(chunks) takes every Chunk produced by Phase 2
(across all filings), embeds dense vectors, fits+encodes BM25 sparse vectors,
and upserts both into one Qdrant collection.
WHY one call across ALL chunks rather than per-filing: BM25 IDF is only
meaningful relative to the whole corpus (see sparse_indexer.py) — fitting
per filing would make every collection-wide retrieval comparison invalid.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.indexing.dense_indexer import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    upsert_points,
)
from src.indexing.embeddings import BGEEmbedder
from src.indexing.sparse_indexer import BM25SparseEncoder
from src.processing.models import Chunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IndexManager:
    """Wires BGEEmbedder + BM25SparseEncoder + Qdrant collection into one indexing pipeline."""

    def __init__(self, cfg) -> None:
        """Open an embedded (no-Docker) QdrantClient and load the dense embedding model.

        Args:
            cfg: resolved Hydra/OmegaConf config (configs/base.yaml).

        WHY QdrantClient(path=...) instead of host/port: no Docker on this
        machine — embedded local mode persists to disk and needs no server,
        while staying on the same qdrant-client API the server-mode docker
        setup would use, so swapping back to `QdrantClient(url="localhost")`
        later is a one-line change.
        """
        self.cfg = cfg
        self.client = QdrantClient(path=cfg.indexing.qdrant_path)
        self.embedder = BGEEmbedder(cfg.embeddings.dense_model, cfg.embeddings.batch_size)
        self.bm25 = BM25SparseEncoder()

    def build_index(self, chunks: list[Chunk]) -> int:
        """Embed + index every chunk into the Qdrant collection. Returns point count.

        Args:
            chunks: all Chunks across all filings being indexed in this run.

        WHY payload carries the full chunk (text + metadata) rather than
        just an id: retrieval (Phase 5) and generation (Phase 6) need the
        chunk text and its filing identity directly off the search hit —
        round-tripping to disk for every retrieved chunk would add latency
        for no benefit at this corpus size.
        """
        ensure_collection(
            self.client,
            collection_name=self.cfg.indexing.collection_name,
            dense_dim=self.cfg.embeddings.dense_dimensions,
            distance_metric=self.cfg.indexing.distance_metric,
            hnsw_m=self.cfg.indexing.hnsw_m,
            hnsw_ef_construct=self.cfg.indexing.hnsw_ef_construct,
        )

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks (dense)")
        dense_vecs = self.embedder.embed(texts)

        logger.info(f"Fitting BM25 over {len(texts)} chunks (sparse)")
        self.bm25.fit(texts)
        sparse_vecs = self.bm25.encode_all()
        self.bm25.save_vocab(self.cfg.indexing.bm25_vocab_path)

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vecs[i].tolist(),
                    SPARSE_VECTOR_NAME: sparse_vecs[i],
                },
                payload=chunk.model_dump(mode="json"),
            )
            for i, chunk in enumerate(chunks)
        ]
        upsert_points(self.client, self.cfg.indexing.collection_name, points)
        return len(points)
