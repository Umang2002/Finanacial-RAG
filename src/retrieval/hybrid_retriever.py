"""Reciprocal Rank Fusion (RRF) to merge dense + sparse retrieval results into a unified ranking.

WHAT: rrf_fuse() merges any number of ranked RetrievedChunk lists (dense
search on the raw query, dense search on a HyDE doc, dense search on each
multi-query paraphrase, sparse search on each of those too) into one ranking
by rank position, not raw score. HybridRetriever wires DenseRetriever +
SparseRetriever together and runs both over every query variant Phase 4
produced.
WHY RRF over a weighted sum of raw scores: dense cosine similarity and BM25
dot-product scores live on incomparable scales (CLAUDE.md "Key Tech
Choices" — RRF is parameter-free and outperforms weighted sum in practice),
so only each list's *rank order* is combined, never the raw score values.
"""

from __future__ import annotations

from omegaconf import DictConfig
from qdrant_client import QdrantClient

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.models import RetrievedChunk
from src.retrieval.sparse_retriever import SparseRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]], k: int = _DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse N ranked lists into one ranking via Reciprocal Rank Fusion.

    Args:
        ranked_lists: one list per retrieval call (dense/sparse x query variant),
            each already sorted best-first.
        k: RRF damping constant — higher k flattens the gap between rank 1
            and rank N within a single list (standard default is 60).

    Returns:
        Chunks sorted by summed RRF score (desc), each carrying the union of
        `sources` it was retrieved under across every input list.

    WHY chunk_id as the dedup key: the same chunk routinely appears in both
    the dense and sparse lists (and across query variants) — summing its
    1/(k+rank) contribution from every list it appears in is the whole point
    of RRF, so dedup must happen by identity, not by dropping later
    occurrences.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    sources: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(chunk_id, hit)
            sources.setdefault(chunk_id, set()).update(hit.sources)

    fused = [
        RetrievedChunk(chunk=chunks[chunk_id].chunk, score=score, sources=sorted(sources[chunk_id]))
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


class HybridRetriever:
    """Runs dense + sparse search over every query variant, then RRF-fuses the results."""

    def __init__(self, cfg: DictConfig, client: QdrantClient) -> None:
        """Build both retrievers on one shared QdrantClient.

        WHY shared: embedded-mode Qdrant only allows one writer/reader.
        """
        self.cfg = cfg
        self.dense = DenseRetriever(cfg, client)
        self.sparse = SparseRetriever(cfg, client)

    def retrieve(self, queries: list[str]) -> list[RetrievedChunk]:
        """Run dense + sparse search for every query variant and RRF-fuse all resulting lists.

        Args:
            queries: every query variant to search with — the raw query plus
                whatever Phase 4 produced (HyDE doc, multi-query paraphrases,
                decomposed sub-questions). Callers pass `[raw_query]` to skip
                expansion entirely.
        """
        ranked_lists = [self.dense.search(q) for q in queries] + [
            self.sparse.search(q) for q in queries
        ]
        fused = rrf_fuse(ranked_lists)
        logger.debug("Fused %d query variants into %d unique chunks", len(queries), len(fused))
        return fused
