"""Classifies query intent via a local LLM zero-shot prompt — decides downstream transforms to run.

WHAT: QueryAnalyzer.analyze() asks the local LLM to pick one intent label
for a raw query and wraps the result in an AnalyzedQuery.
WHY a classifier at all: HyDE/multi-query help every query, but
decomposition only helps genuinely multi-hop questions ("How did revenue
growth compare between AAPL and MSFT in FY23?") — running it on a simple
lookup ("What was AAPL's FY23 revenue?") just adds latency and noise.
is_multi_hop on AnalyzedQuery is what src/query/query_decomposer.py and
Phase 5's retrieval orchestration gate on.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.query.models import AnalyzedQuery, QueryIntent
from src.utils.llm_client import OllamaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

_INTENTS: tuple[QueryIntent, ...] = (
    "factual_lookup",
    "comparison",
    "calculation",
    "multi_hop",
    "definition",
    "other",
)

_SYSTEM_PROMPT = """You classify questions about SEC financial filings (10-K/10-Q) into \
exactly one intent label. Respond with ONLY the label, nothing else.

Labels:
- factual_lookup: asks for a single fact directly stated in a filing \
(e.g. "What was Apple's FY2023 revenue?")
- comparison: asks to compare two or more entities, years, or metrics \
(e.g. "How did Apple's margin compare to Microsoft's?")
- calculation: requires arithmetic over filing data \
(e.g. "What was the YoY growth rate?")
- multi_hop: requires combining facts from multiple distinct parts of one \
or more filings to answer \
(e.g. "Which segment grew fastest and what drove that growth?")
- definition: asks what a financial term or line item means \
(e.g. "What does Apple count as an operating lease?")
- other: anything that doesn't fit the above

Respond with exactly one label from: factual_lookup, comparison, \
calculation, multi_hop, definition, other"""

_USER_PROMPT_TEMPLATE = "Question: {query}\nLabel:"


class QueryAnalyzer:
    """Wraps an LLM call that maps a raw query to one of `_INTENTS`."""

    def __init__(self, cfg: DictConfig, llm: OllamaClient | None = None) -> None:
        """Build the analyzer; reuse injected `llm` in tests, else build from cfg.generation."""
        self.cfg = cfg
        self.llm = llm or OllamaClient(model=cfg.generation.model, host=cfg.generation.ollama_host)

    def analyze(self, query: str) -> AnalyzedQuery:
        """Classify `query` into one of `_INTENTS` and flag multi-hop questions."""
        raw_response = self.llm.complete(
            _USER_PROMPT_TEMPLATE.format(query=query),
            system=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        intent = _parse_intent(raw_response)
        logger.debug("Classified query intent=%s for query=%r", intent, query)
        return AnalyzedQuery(raw_query=query, intent=intent, is_multi_hop=intent == "multi_hop")


def _parse_intent(raw_response: str) -> QueryIntent:
    """Extract a known intent label from free-text LLM output, defaulting to "other".

    WHY substring match instead of exact match: local 3B models sometimes
    wrap the label in a sentence ("The label is: comparison.") despite the
    system prompt asking for just the label — matching the first known
    label that appears anywhere in the response is more robust than
    requiring an exact match and failing closed to "other".
    """
    cleaned = raw_response.strip().lower()
    for intent in _INTENTS:
        if intent in cleaned:
            return intent
    return "other"
