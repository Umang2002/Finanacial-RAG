"""Unit tests for src/evaluation/ragas_evaluator.py.

WHY patch ragas.evaluate() rather than fake a BaseRagasLLM: RAGAS metrics
drive the injected LLM through an internal structured-generation protocol
(per-metric prompt schemas) — faking that faithfully would reimplement more
of RAGAS than the test is worth. Injecting sentinel llm/embeddings objects
(never touched, since evaluate() itself is mocked) verifies RagasEvaluator
wires SingleTurnSample/EvaluationDataset correctly and that NaN scores
degrade to None, without a real Ollama call.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from src.evaluation.ragas_evaluator import RagasEvaluator, _clean


def _cfg() -> OmegaConf:
    return OmegaConf.create({"generation": {"model": "llama3.2:3b", "ollama_host": "x"}})


def test_clean_passes_through_real_score() -> None:
    assert _clean(0.75) == 0.75


def test_clean_converts_nan_to_none() -> None:
    assert _clean(math.nan) is None


def test_clean_passes_through_none() -> None:
    assert _clean(None) is None


def test_evaluate_extracts_all_four_metrics_from_result() -> None:
    fake_llm = MagicMock()
    fake_embeddings = MagicMock()
    evaluator = RagasEvaluator(_cfg(), llm=fake_llm, embeddings=fake_embeddings)

    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame(
        [
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": math.nan,
                "context_recall": 0.7,
            }
        ]
    )

    with patch(
        "src.evaluation.ragas_evaluator.evaluate", return_value=fake_result
    ) as mock_evaluate:
        result = evaluator.evaluate(
            "fb_001", "What were net sales?", "Net sales were $X [1].", ["Net sales were $X."], "$X"
        )

    assert result.financebench_id == "fb_001"
    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.8
    assert result.context_precision is None
    assert result.context_recall == 0.7

    call_kwargs = mock_evaluate.call_args.kwargs
    assert call_kwargs["llm"] is fake_llm
    assert call_kwargs["embeddings"] is fake_embeddings
    assert call_kwargs["raise_exceptions"] is False
    sample = call_kwargs["dataset"].samples[0]
    assert sample.user_input == "What were net sales?"
    assert sample.response == "Net sales were $X [1]."
    assert sample.reference == "$X"
