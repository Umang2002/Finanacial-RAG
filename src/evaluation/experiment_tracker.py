"""Logs experiment configs and results for ablation study comparisons.

WHAT: log_experiment() appends one EvalReport (plus the config knobs that
produced it) as one line to a JSONL file. WHY JSONL, not a single JSON
array: CLAUDE.md's experiment workflow is "one experiment = one config
override in configs/experiments/" run repeatedly over time — append-only
avoids read-modify-write races and a parse error in one run can't corrupt
every prior run's record.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.evaluation.models import EvalReport
from src.utils.logging import get_logger

logger = get_logger(__name__)

_RELEVANT_CONFIG_KEYS = ("chunking", "retrieval", "query", "generation")


def log_experiment(cfg: DictConfig, report: EvalReport, log_path: str | Path) -> None:
    """Append one experiment's config knobs + aggregate metrics as a JSONL row.

    Args:
        cfg: the resolved run config — only the sections that plausibly
            affect retrieval/generation quality are recorded (not the
            whole tree) so the log stays readable across many runs.
        report: this run's aggregate EvalReport.
        log_path: e.g. data/eval/experiment_log.jsonl.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "config_name": report.config_name,
        "num_examples": report.num_examples,
        "config": {
            key: OmegaConf.to_container(cfg[key]) for key in _RELEVANT_CONFIG_KEYS if key in cfg
        },
        "metrics": {
            "hit_rate": report.mean_hit_rate,
            "mrr": report.mean_mrr,
            "ndcg": report.mean_ndcg,
            "precision_at_k": report.mean_precision_at_k,
            "recall_at_k": report.mean_recall_at_k,
            "faithfulness": report.mean_faithfulness,
            "answer_relevancy": report.mean_answer_relevancy,
            "context_precision": report.mean_context_precision,
            "context_recall": report.mean_context_recall,
        },
    }

    with log_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info(f"Logged experiment '{report.config_name}' to {log_path}")


def load_experiments(log_path: str | Path) -> list[dict]:
    """Read every logged experiment row back, in run order — for the eval comparison table."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
