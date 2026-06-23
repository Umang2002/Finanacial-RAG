"""Approximate nearest-neighbor search over Qdrant dense vector index.

WHAT: DenseRetriever.search() embeds a query string with BGEEmbedder and runs
ANN search against the `dense` named vector in the `financial_rag` collection.
WHY a thin class around one Qdrant call: same seam pattern as
src/indexing/index_manager.py — keeps BGEEmbedder model loading in one place
and gives hybrid_retriever.py/tests a single thing to mock.
"""

from __future__ import annotations

from omegaconf import DictConfig
from qdrant_client import QdrantClient

from src.indexing.dense_indexer import DENSE_VECTOR_NAME
from src.indexing.embeddings import BGEEmbedder
from src.processing.models import Chunk
from src.retrieval.models import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Embeds a query and runs ANN search against the `dense` named vector."""

    def __init__(
        self, cfg: DictConfig, client: QdrantClient, embedder: BGEEmbedder | None = None
    ) -> None:
        """Reuse a caller-owned QdrantClient (shared with SparseRetriever) and a BGEEmbedder.

        WHY client is injected, not opened here: embedded-mode Qdrant locks
        its storage directory to one process — RetrievalPipeline opens a
        single QdrantClient and hands it to both DenseRetriever and
        SparseRetriever instead of each opening its own.
        """
        self.cfg = cfg
        self.client = client
        self.embedder = embedder or BGEEmbedder(
            cfg.embeddings.dense_model, cfg.embeddings.batch_size
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed `query` and return the top_k nearest chunks by cosine similarity.

        Args:
            query: raw query text or a query variant (HyDE doc, paraphrase).
            top_k: defaults to cfg.retrieval.top_k_dense.
        """
        top_k = top_k if top_k is not None else self.cfg.retrieval.top_k_dense
        query_vector = self.embedder.embed([query])[0]

        response = self.client.query_points(
            collection_name=self.cfg.indexing.collection_name,
            query=query_vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=Chunk.model_validate(point.payload), score=point.score, sources=["dense"]
            )
            for point in response.points
        ]
