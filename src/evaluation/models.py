"""Pydantic models for evaluation — input is Phase 6's GeneratedAnswer, output feeds the CLI report.

WHAT: EvalExample is one FinanceBench question + its ground-truth answer and
source filing. RetrievalEvalResult/RagasEvalResult hold the per-example
metric scores from src/evaluation/retrieval_metrics.py and ragas_evaluator.py
respectively. EvalReport aggregates every example into the run-level numbers
printed by scripts/run_eval.py and persisted by experiment_tracker.py.
WHY pydantic: crosses the eval dataset -> run_eval.py -> experiment_tracker.py
module boundary, per CLAUDE.md "use pydantic for data structures that cross
module boundaries".
"""

from __future__ import annotations

from pydantic import BaseModel


class EvalExample(BaseModel):
    """One FinanceBench question, normalized to this project's local corpus identity.

    WHY doc_name is kept verbatim from FinanceBench (e.g. "MICROSOFT_2023_10K")
    alongside ticker/fiscal_year: it's the join key back to FinanceBench's own
    evidence text, while ticker/fiscal_year is what the local pipeline's
    ChunkMetadata uses — both are needed, they aren't interchangeable.
    """

    financebench_id: str
    ticker: str
    fiscal_year: int
    doc_name: str
    question: str
    ground_truth_answer: str
    evidence_texts: list[str]


class RetrievalEvalResult(BaseModel):
    """Retrieval-only metrics for one example — relevance judged by evidence-text overlap."""

    financebench_id: str
    hit_rate: float
    mrr: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float


class RagasEvalResult(BaseModel):
    """RAGAS LLM-judged metrics for one example — None on error, not skipped silently."""

    financebench_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


class EvalReport(BaseModel):
    """Run-level aggregate: mean of every metric across examples, plus the raw per-example rows."""

    config_name: str
    num_examples: int
    retrieval_results: list[RetrievalEvalResult]
    ragas_results: list[RagasEvalResult]
    mean_hit_rate: float
    mean_mrr: float
    mean_ndcg: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_faithfulness: float | None
    mean_answer_relevancy: float | None
    mean_context_precision: float | None
    mean_context_recall: float | None
