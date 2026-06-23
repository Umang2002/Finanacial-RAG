"""RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall.

WHAT: RagasEvaluator.evaluate() scores one (query, answer, retrieved
contexts, ground-truth answer) tuple with RAGAS's four LLM-judged metrics.
WHY ChatGroq + local BGE-M3 instead of RAGAS's OpenAI default: `ChatGroq`
reuses the same hosted model Phase 6 generation uses as the judge LLM
(fast, avoids the local-model judge timeouts seen running this on Ollama),
and `_BGERagasEmbeddings` reuses Phase 3's already-loaded local BGE-M3
model for `answer_relevancy`'s embedding similarity step instead of an
API embedding call.
"""

from __future__ import annotations

import asyncio

from langchain_groq import ChatGroq
from omegaconf import DictConfig
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.evaluation.models import RagasEvalResult
from src.indexing.embeddings import BGEEmbedder
from src.utils.logging import get_logger

logger = get_logger(__name__)

_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


class _BGERagasEmbeddings(BaseRagasEmbeddings):
    """Adapts BGEEmbedder (sync, batched) to RAGAS's async embeddings interface.

    WHY run_in_executor instead of a real async embed call: BGEEmbedder
    wraps a local sentence-transformers model — there's no I/O to await,
    only CPU/GPU compute, so the only way to not block RAGAS's asyncio
    event loop is to push the sync call onto a worker thread.
    """

    def __init__(self, embedder: BGEEmbedder) -> None:
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_documents, texts)


class RagasEvaluator:
    """Wires a ChatGroq judge + local BGE-M3 embeddings into RAGAS's evaluate() call."""

    def __init__(
        self,
        cfg: DictConfig,
        llm: LangchainLLMWrapper | None = None,
        embeddings: BaseRagasEmbeddings | None = None,
    ) -> None:
        """Build the judge LLM/embeddings from cfg, or reuse injected ones in tests."""
        self.llm = llm or LangchainLLMWrapper(
            ChatGroq(
                model=cfg.generation.model,
                api_key=cfg.generation.groq_api_key,
                temperature=0.0,
            )
        )
        self.embeddings = embeddings or _BGERagasEmbeddings(
            BGEEmbedder(cfg.embeddings.dense_model, cfg.embeddings.batch_size)
        )

    def evaluate(
        self,
        financebench_id: str,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        reference_answer: str,
    ) -> RagasEvalResult:
        """Score one example with all four RAGAS metrics; a failed metric is None, not a crash.

        WHY raise_exceptions=False: a judge call can still time out or
        produce unparsable output for one metric on one example — that
        should degrade that single score to None, not abort the whole
        eval run.
        """
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                    reference=reference_answer,
                )
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=_METRICS,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
            show_progress=False,
        )
        scores = result.to_pandas().iloc[0]
        logger.debug(f"RAGAS scores for {financebench_id}: {scores.to_dict()}")
        return RagasEvalResult(
            financebench_id=financebench_id,
            faithfulness=_clean(scores.get("faithfulness")),
            answer_relevancy=_clean(scores.get("answer_relevancy")),
            context_precision=_clean(scores.get("context_precision")),
            context_recall=_clean(scores.get("context_recall")),
        )


def _clean(value: object) -> float | None:
    """RAGAS reports a failed metric as NaN — surface that as None so callers don't average NaN."""
    if value is None:
        return None
    score = float(value)  # type: ignore[arg-type]
    return None if score != score else score  # NaN != NaN
