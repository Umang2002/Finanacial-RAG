"""Unit tests for src/evaluation/retrieval_metrics.py."""

from __future__ import annotations

from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    judge_relevance,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_judge_relevance_flags_chunk_contained_in_evidence() -> None:
    chunks = ["Net sales were $383.3 billion in fiscal 2023.", "Unrelated paragraph about hiring."]
    evidence = ["Apple reported net sales were $383.3 billion in fiscal 2023, up slightly."]

    relevant = judge_relevance(chunks, evidence)

    assert relevant == [True, False]


def test_judge_relevance_respects_containment_threshold() -> None:
    chunks = ["completely different unrelated text about something else entirely"]
    evidence = ["net sales were $383.3 billion"]

    assert judge_relevance(chunks, evidence) == [False]


def test_judge_relevance_empty_chunk_is_not_relevant() -> None:
    assert judge_relevance([""], ["net sales were $383.3 billion"]) == [False]


def test_hit_rate_at_k_true_if_any_relevant_in_top_k() -> None:
    assert hit_rate_at_k([False, True, False], k=2) == 1.0
    assert hit_rate_at_k([False, False, True], k=2) == 0.0


def test_mrr_reciprocal_rank_of_first_relevant() -> None:
    assert mrr([False, True, True]) == 0.5
    assert mrr([True, False]) == 1.0
    assert mrr([False, False]) == 0.0


def test_precision_at_k_fraction_relevant_in_top_k() -> None:
    assert precision_at_k([True, True, False, False], k=2) == 1.0
    assert precision_at_k([True, False, False, False], k=4) == 0.25
    assert precision_at_k([], k=3) == 0.0


def test_recall_at_k_fraction_of_total_relevant_captured() -> None:
    assert recall_at_k([True, False, True], k=3, total_relevant=4) == 0.5
    assert recall_at_k([True], k=1, total_relevant=0) == 0.0


def test_ndcg_at_k_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k([True, True, False], k=3) == 1.0


def test_ndcg_at_k_no_relevant_scores_zero() -> None:
    assert ndcg_at_k([False, False], k=2) == 0.0


def test_ndcg_at_k_worse_ranking_scores_below_one() -> None:
    score = ndcg_at_k([False, True], k=2)
    assert 0.0 < score < 1.0
