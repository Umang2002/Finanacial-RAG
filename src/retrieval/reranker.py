"""Cross-encoder reranking: BGE-reranker-v2-m3 (local, free) over RRF-fused candidates.

WHAT: Reranker.rerank() scores every (query, chunk) pair with a cross-encoder
and returns the top_k_rerank chunks by that score, discarding the rest.
WHY a second model after RRF fusion: dense/sparse retrieval optimize for
recall over the whole corpus via cheap bi-encoder/BM25 scoring — a
cross-encoder that jointly attends over the query and each candidate's full
text is far more accurate at judging relevance, but too slow to run over
the entire collection, so it only reranks the small fused candidate set.
"""

from __future__ import annotations

from omegaconf import DictConfig
from sentence_transformers import CrossEncoder

from src.retrieval.models import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

_RERANKER_MODELS = {
    "bge": "BAAI/bge-reranker-v2-m3",
}


class Reranker:
    """Loads a local cross-encoder once and reranks a candidate list per query."""

    def __init__(self, cfg: DictConfig, model: CrossEncoder | None = None) -> None:
        """Load the cross-encoder named by cfg.retrieval.reranker (only "bge" is free/local)."""
        self.cfg = cfg
        model_name = _RERANKER_MODELS[cfg.retrieval.reranker]
        logger.info(f"Loading reranker model: {model_name}")
        self.model = model or CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Score every candidate against `query` and return the top_k by cross-encoder score.

        Args:
            query: the raw user query — always the original question, never
                a HyDE doc or paraphrase, since reranking judges relevance
                to what the user actually asked.
            candidates: RRF-fused hits from HybridRetriever.retrieve().
            top_k: defaults to cfg.retrieval.top_k_rerank.

        WHY score replaces the RRF score on the returned RetrievedChunk
        rather than stacking alongside it: once reranking has run, the
        cross-encoder score is the most relevance-accurate signal available
        and is what downstream consumers (citation ordering, eval metrics)
        should sort/threshold on — the RRF score's job (deciding the
        candidate set) is already done.
        """
        if not candidates:
            return []
        top_k = top_k if top_k is not None else self.cfg.retrieval.top_k_rerank

        pairs = [(query, hit.chunk.text) for hit in candidates]
        ce_scores = self.model.predict(pairs)

        reranked = [
            RetrievedChunk(chunk=hit.chunk, score=float(score), sources=hit.sources)
            for hit, score in zip(candidates, ce_scores)
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked[:top_k]
