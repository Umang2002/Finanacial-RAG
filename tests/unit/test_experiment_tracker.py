"""Unit tests for src/evaluation/experiment_tracker.py."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.evaluation.experiment_tracker import load_experiments, log_experiment
from src.evaluation.models import EvalReport


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 512},
            "retrieval": {"top_k_rerank": 5},
        }
    )


def _report(config_name: str = "baseline") -> EvalReport:
    return EvalReport(
        config_name=config_name,
        num_examples=1,
        retrieval_results=[],
        ragas_results=[],
        mean_hit_rate=1.0,
        mean_mrr=1.0,
        mean_ndcg=1.0,
        mean_precision_at_k=0.5,
        mean_recall_at_k=0.5,
        mean_faithfulness=0.9,
        mean_answer_relevancy=0.8,
        mean_context_precision=None,
        mean_context_recall=None,
    )


def test_log_experiment_appends_one_jsonl_row(tmp_path: Path) -> None:
    log_path = tmp_path / "experiment_log.jsonl"

    log_experiment(_cfg(), _report(), log_path)

    rows = load_experiments(log_path)
    assert len(rows) == 1
    assert rows[0]["config_name"] == "baseline"
    assert rows[0]["config"]["chunking"]["chunk_size"] == 512
    assert rows[0]["metrics"]["faithfulness"] == 0.9
    assert rows[0]["metrics"]["context_precision"] is None


def test_log_experiment_appends_without_overwriting_prior_runs(tmp_path: Path) -> None:
    log_path = tmp_path / "experiment_log.jsonl"

    log_experiment(_cfg(), _report("run_a"), log_path)
    log_experiment(_cfg(), _report("run_b"), log_path)

    rows = load_experiments(log_path)
    assert [r["config_name"] for r in rows] == ["run_a", "run_b"]


def test_load_experiments_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert load_experiments(tmp_path / "nonexistent.jsonl") == []


def test_log_experiment_redacts_api_key_in_generation_config(tmp_path: Path) -> None:
    log_path = tmp_path / "experiment_log.jsonl"
    cfg = OmegaConf.create(
        {"generation": {"model": "llama-3.3-70b-versatile", "groq_api_key": "gsk_realsecret123"}}
    )

    log_experiment(cfg, _report(), log_path)

    rows = load_experiments(log_path)
    assert rows[0]["config"]["generation"]["groq_api_key"] == "***REDACTED***"
    assert rows[0]["config"]["generation"]["model"] == "llama-3.3-70b-versatile"
