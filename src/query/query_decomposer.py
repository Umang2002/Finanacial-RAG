"""Breaks multi-hop questions into sub-questions, each retrievable on its own.

WHAT: QueryDecomposer.decompose() asks the LLM to split a question into N
independently-answerable sub-questions; Phase 5 retrieves for each
sub-question separately and Phase 6 combines the retrieved context.
WHY only for multi-hop: a question like "Which segment grew fastest and
what drove that growth?" needs two different lookups (growth-rate
comparison, then a causal explanation for the winner) that don't share a
single relevant passage — retrieving against the combined question tends
to surface chunks that are weakly relevant to both halves instead of
strongly relevant to either. Single-hop questions decompose to themselves
(a one-element list) so callers don't need a separate "skip decomposition"
branch.
Gated by `cfg.query.use_decomposition` — see configs/base.yaml comment
"Enable for multi-hop questions".
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.query.models import DecomposedQuery
from src.utils.llm_client import OllamaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You break a multi-hop financial question into the smallest number of \
independently-answerable sub-questions needed to answer it. Each sub-question must be \
answerable on its own from a single passage of a SEC filing. Respond with one sub-question \
per line, no numbering, no extra commentary. If the question is already a single fact \
lookup, respond with just that question unchanged."""

_USER_PROMPT_TEMPLATE = "Question: {query}\nSub-questions:"


class QueryDecomposer:
    """Wraps the LLM call behind multi-hop question decomposition."""

    def __init__(self, cfg: DictConfig, llm: OllamaClient | None = None) -> None:
        """Build the decomposer; reuse injected `llm` in tests, else build from cfg.generation."""
        self.cfg = cfg
        self.llm = llm or OllamaClient(model=cfg.generation.model, host=cfg.generation.ollama_host)

    def decompose(self, query: str) -> DecomposedQuery:
        """Split `query` if `cfg.query.use_decomposition` is set, else return it unchanged."""
        if not self.cfg.query.use_decomposition:
            return DecomposedQuery(raw_query=query, sub_questions=[query])

        raw_response = self.llm.complete(
            _USER_PROMPT_TEMPLATE.format(query=query),
            system=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        sub_questions = [line.strip(" -\t") for line in raw_response.splitlines() if line.strip()]
        if not sub_questions:
            sub_questions = [query]
        logger.debug("Decomposed query=%r into %d sub-questions", query, len(sub_questions))
        return DecomposedQuery(raw_query=query, sub_questions=sub_questions)
