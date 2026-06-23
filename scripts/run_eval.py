"""CLI: runs RAGAS + retrieval evaluation against the FinanceBench dataset.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --limit 5 --config-name smoke_test

WHY a thin CLI: real logic lives in src/evaluation/*.py (same pattern as
scripts/generate.py for Phase 6). This script only wires config -> loads
data/eval/financebench.json (built by scripts/build_eval_dataset.py) -> runs
each example through the full Phase 1-6 query/retrieve/generate pipeline ->
scores it with retrieval_metrics.py + RagasEvaluator -> aggregates ->
prints a rich summary table -> logs the run via experiment_tracker.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.evaluation.experiment_tracker import log_experiment
from src.evaluation.models import EvalExample, EvalReport, RagasEvalResult, RetrievalEvalResult
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    judge_relevance,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.generation.generator import Generator
from src.query.query_analyzer import QueryAnalyzer
from src.query.query_decomposer import QueryDecomposer
from src.query.query_transformer import QueryTransformer
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides: dataset limit (for smoke tests), config name, config path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N examples (smoke test)"
    )
    parser.add_argument(
        "--config-name",
        default="baseline",
        help="Label for this run in the experiment log (default: baseline)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML (default: configs/base.yaml)"
    )
    return parser.parse_args()


def _mean(values: list[float | None]) -> float | None:
    """Average the non-None scores; None if every score failed (0.0 would understate failure)."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> None:
    """Load config + eval set, run the full pipeline per example, aggregate, print, log."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    examples_raw = json.loads(Path(cfg.evaluation.dataset).read_text())
    examples = [EvalExample.model_validate(e) for e in examples_raw]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        logger.warning(f"No eval examples found at {cfg.evaluation.dataset} — nothing to run")
        return
    logger.info(f"Evaluating {len(examples)} example(s)")

    analyzer = QueryAnalyzer(cfg)
    transformer = QueryTransformer(cfg)
    decomposer = QueryDecomposer(cfg)
    retrieval_pipeline = RetrievalPipeline(cfg)
    generator = Generator(cfg)
    ragas_evaluator = RagasEvaluator(cfg)

    top_k = cfg.retrieval.top_k_rerank
    retrieval_results: list[RetrievalEvalResult] = []
    ragas_results: list[RagasEvalResult] = []

    for example in examples:
        logger.info(f"[{example.financebench_id}] {example.question}")
        analyzed = analyzer.analyze(example.question)
        transformed = transformer.transform(example.question)
        decomposed = decomposer.decompose(example.question) if analyzed.is_multi_hop else None
        chunks = retrieval_pipeline.retrieve(
            example.question, transformed=transformed, decomposed=decomposed
        )
        chunk_texts = [c.chunk.text for c in chunks]

        relevant = judge_relevance(chunk_texts, example.evidence_texts)
        retrieval_results.append(
            RetrievalEvalResult(
                financebench_id=example.financebench_id,
                hit_rate=hit_rate_at_k(relevant, top_k),
                mrr=mrr(relevant),
                ndcg=ndcg_at_k(relevant, top_k),
                precision_at_k=precision_at_k(relevant, top_k),
                recall_at_k=recall_at_k(
                    relevant, top_k, total_relevant=len(example.evidence_texts)
                ),
            )
        )

        generated = generator.generate(example.question, chunks)
        ragas_results.append(
            ragas_evaluator.evaluate(
                example.financebench_id,
                example.question,
                generated.answer,
                chunk_texts,
                example.ground_truth_answer,
            )
        )

    report = EvalReport(
        config_name=args.config_name,
        num_examples=len(examples),
        retrieval_results=retrieval_results,
        ragas_results=ragas_results,
        mean_hit_rate=_mean([r.hit_rate for r in retrieval_results]) or 0.0,
        mean_mrr=_mean([r.mrr for r in retrieval_results]) or 0.0,
        mean_ndcg=_mean([r.ndcg for r in retrieval_results]) or 0.0,
        mean_precision_at_k=_mean([r.precision_at_k for r in retrieval_results]) or 0.0,
        mean_recall_at_k=_mean([r.recall_at_k for r in retrieval_results]) or 0.0,
        mean_faithfulness=_mean([r.faithfulness for r in ragas_results]),
        mean_answer_relevancy=_mean([r.answer_relevancy for r in ragas_results]),
        mean_context_precision=_mean([r.context_precision for r in ragas_results]),
        mean_context_recall=_mean([r.context_recall for r in ragas_results]),
    )

    table = Table(title=f"Eval Report — {report.config_name} (n={report.num_examples})")
    table.add_column("Metric")
    table.add_column("Score")
    for label, value in [
        ("Hit Rate@k", report.mean_hit_rate),
        ("MRR", report.mean_mrr),
        ("NDCG@k", report.mean_ndcg),
        ("Precision@k", report.mean_precision_at_k),
        ("Recall@k", report.mean_recall_at_k),
        ("Faithfulness", report.mean_faithfulness),
        ("Answer Relevancy", report.mean_answer_relevancy),
        ("Context Precision", report.mean_context_precision),
        ("Context Recall", report.mean_context_recall),
    ]:
        table.add_row(label, f"{value:.3f}" if value is not None else "N/A")
    console.print(table)

    results_dir = Path(cfg.evaluation.dataset).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"{report.config_name}.json"
    report_path.write_text(report.model_dump_json(indent=2))
    logger.info(f"Wrote full report to {report_path}")

    log_experiment(cfg, report, Path(cfg.evaluation.dataset).parent / "experiment_log.jsonl")


if __name__ == "__main__":
    main()
