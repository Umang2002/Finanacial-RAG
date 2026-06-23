"""Orchestrates dense + sparse + RRF fusion + reranking; single entry point for Phase 5.

WHAT: RetrievalPipeline.retrieve() takes a raw query plus whatever Phase 4
query processing produced for it (TransformedQuery's HyDE doc/paraphrases,
DecomposedQuery's sub-questions), runs every variant through dense + sparse
search, RRF-fuses the results, and reranks down to the final candidate set
handed to Phase 6 generation.
WHY query variants are passed in rather than generated here: Phase 4
(src/query/*.py) already owns HyDE/multi-query/decomposition and each needs
an LLM call — RetrievalPipeline only does retrieval, so it stays testable
without spinning up an Ollama client, same separation IndexManager keeps
from the chunking phase that feeds it.
"""

from __future__ import annotations

from omegaconf import DictConfig
from qdrant_client import QdrantClient

from src.query.models import DecomposedQuery, TransformedQuery
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import Reranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_query_variants(
    raw_query: str,
    transformed: TransformedQuery | None = None,
    decomposed: DecomposedQuery | None = None,
) -> list[str]:
    """Collect every distinct query string to retrieve with, raw query first.

    WHY dedupe while preserving order: DecomposedQuery.sub_questions is
    `[raw_query]` when decomposition is off/skipped (see
    src/query/query_decomposer.py) — without dedup the raw query would be
    searched twice and double-count in RRF fusion.
    """
    variants = [raw_query]
    if transformed is not None:
        if transformed.hyde_doc:
            variants.append(transformed.hyde_doc)
        variants.extend(transformed.multi_queries)
    if decomposed is not None:
        variants.extend(decomposed.sub_questions)

    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


class RetrievalPipeline:
    """Wires HybridRetriever (dense + sparse + RRF) and Reranker into one retrieval call."""

    def __init__(self, cfg: DictConfig, client: QdrantClient | None = None) -> None:
        """Open an embedded QdrantClient (or reuse an injected one in tests) and load both models.

        WHY client is optional+injectable: same pattern as
        IndexManager/DenseRetriever — tests redirect this to an in-memory
        QdrantClient instead of touching the real on-disk collection.
        """
        self.cfg = cfg
        self.client = client or QdrantClient(path=cfg.indexing.qdrant_path)
        self.hybrid = HybridRetriever(cfg, self.client)
        self.reranker = Reranker(cfg)

    def retrieve(
        self,
        raw_query: str,
        transformed: TransformedQuery | None = None,
        decomposed: DecomposedQuery | None = None,
    ) -> list[RetrievedChunk]:
        """Run the full Phase 5 pipeline: expand -> dense+sparse search -> RRF fuse -> rerank.

        Args:
            raw_query: the original user question — always what reranking
                scores against, regardless of which variants found a chunk.
            transformed: Phase 4 HyDE/multi-query output, or None to search
                with only the raw query on the dense/sparse side.
            decomposed: Phase 4 decomposition output, or None to skip it.

        Returns:
            Final top_k_rerank chunks, cross-encoder-scored, ready for
            Phase 6 context assembly.
        """
        variants = build_query_variants(raw_query, transformed, decomposed)
        logger.info(f"Retrieving for query={raw_query!r} with {len(variants)} variant(s)")
        fused = self.hybrid.retrieve(variants)
        return self.reranker.rerank(raw_query, fused)
