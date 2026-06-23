"""Retrieval evaluation metrics: MRR, NDCG, Hit Rate, Precision@k, Recall@k.

WHAT: every metric function takes `relevant: list[bool]` — one flag per
retrieved rank, already-truncated to whatever k the caller wants evaluated.
`judge_relevance()` is the one FinanceBench-specific piece: it produces that
boolean list from chunk text vs. FinanceBench's gold `evidence_text`, since
FinanceBench has no chunk-level ground truth, only page-level evidence text.
WHY binary relevance, not graded: FinanceBench evidence is "this passage
either contains the answer or it doesn't" — there's no graded relevance
judgment in the dataset to build graded NDCG from.
"""

from __future__ import annotations

import math
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, alnum-only word set — punctuation/whitespace shouldn't break overlap matching."""
    return set(_WORD_RE.findall(text.lower()))


def judge_relevance(
    chunk_texts: list[str], evidence_texts: list[str], containment_threshold: float = 0.5
) -> list[bool]:
    """Flag each retrieved chunk as relevant if most of its words appear in some gold evidence text.

    WHY containment (chunk tokens found in evidence) rather than Jaccard:
    a chunk is a short fragment of the much longer `evidence_text_full_page`
    FinanceBench ships — Jaccard would be dominated by the page's extra
    tokens and always score low even for a perfectly on-target chunk.
    """
    evidence_token_sets = [_tokenize(e) for e in evidence_texts]
    relevant = []
    for chunk_text in chunk_texts:
        chunk_tokens = _tokenize(chunk_text)
        if not chunk_tokens:
            relevant.append(False)
            continue
        is_relevant = any(
            len(chunk_tokens & evidence_tokens) / len(chunk_tokens) >= containment_threshold
            for evidence_tokens in evidence_token_sets
        )
        relevant.append(is_relevant)
    return relevant


def hit_rate_at_k(relevant: list[bool], k: int) -> float:
    """1.0 if any of the top-k retrieved chunks is relevant, else 0.0."""
    return 1.0 if any(relevant[:k]) else 0.0


def mrr(relevant: list[bool]) -> float:
    """Reciprocal rank of the first relevant chunk (1-indexed), 0.0 if none found."""
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant."""
    top_k = relevant[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def recall_at_k(relevant: list[bool], k: int, total_relevant: int) -> float:
    """Fraction of all known-relevant chunks captured within the top-k.

    WHY total_relevant is a separate arg rather than derived from `relevant`:
    `relevant` only covers the chunks actually retrieved (top_k_rerank from
    Phase 5) — the true denominator is however many relevant chunks exist in
    the whole corpus, which the caller must supply (e.g. len(evidence_texts)
    as a lower-bound proxy, since FinanceBench doesn't enumerate every
    relevant passage either).
    """
    if total_relevant <= 0:
        return 0.0
    return sum(relevant[:k]) / total_relevant


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    """Binary-relevance NDCG@k: DCG of the actual ranking over DCG of the ideal ranking."""
    top_k = relevant[:k]
    if not any(top_k):
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(top_k, start=1) if is_relevant
    )
    ideal = sorted(top_k, reverse=True)
    idcg = sum(
        1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(ideal, start=1) if is_relevant
    )
    return dcg / idcg if idcg > 0 else 0.0
