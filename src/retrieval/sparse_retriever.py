"""BM25 keyword search against the sparse vector index in Qdrant.

WHAT: SparseRetriever.search() encodes a query into a Qdrant SparseVector
using the persisted BM25 vocab+idf (written by Phase 3's
src/indexing/sparse_indexer.py) and runs sparse search against the `sparse`
named vector in the `financial_rag` collection.
WHY load_vocab() not fit(): the query-time vocab/idf must match what was
indexed exactly — refitting against an empty or different corpus at query
time would silently produce a different term->id mapping than the one
baked into every indexed document's sparse vector.
"""

from __future__ import annotations

from omegaconf import DictConfig
from qdrant_client import QdrantClient

from src.indexing.dense_indexer import SPARSE_VECTOR_NAME
from src.indexing.sparse_indexer import BM25SparseEncoder
from src.processing.models import Chunk
from src.retrieval.models import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """Encodes a query via the persisted BM25 vocab and runs sparse search."""

    def __init__(
        self, cfg: DictConfig, client: QdrantClient, encoder: BM25SparseEncoder | None = None
    ) -> None:
        """Reuse a caller-owned QdrantClient and load the persisted BM25 vocab+idf."""
        self.cfg = cfg
        self.client = client
        self.encoder = encoder or _load_encoder(cfg.indexing.bm25_vocab_path)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Encode `query` as a BM25 sparse vector and return the top_k keyword matches.

        Args:
            query: raw query text or a query variant (HyDE doc, paraphrase).
            top_k: defaults to cfg.retrieval.top_k_sparse.
        """
        top_k = top_k if top_k is not None else self.cfg.retrieval.top_k_sparse
        sparse_vector = self.encoder.encode_query(query)
        if not sparse_vector.indices:
            logger.debug("Query=%r has no in-vocab terms — sparse search returns no hits", query)
            return []

        response = self.client.query_points(
            collection_name=self.cfg.indexing.collection_name,
            query=sparse_vector,
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=Chunk.model_validate(point.payload), score=point.score, sources=["sparse"]
            )
            for point in response.points
        ]


def _load_encoder(vocab_path: str) -> BM25SparseEncoder:
    """Build a BM25SparseEncoder pre-loaded with the persisted vocab+idf."""
    encoder = BM25SparseEncoder()
    encoder.load_vocab(vocab_path)
    return encoder
