"""HyDE expansion and multi-query generation — widens the net before dense/sparse retrieval runs.

WHAT: QueryTransformer.transform() optionally generates a HyDE hypothetical
document (embedded instead of/alongside the raw query for dense search) and
N paraphrased queries (each run through retrieval separately, then fused).
WHY HyDE: a short question and the dense passage that answers it often sit
far apart in embedding space ("What was Apple's FY23 revenue?" vs. a 10-K
paragraph reporting "$383.3 billion in net sales"); embedding a hypothetical
*answer* instead of the question closes that gap.
WHY multi-query: a single query phrasing can miss a chunk that uses
different wording for the same concept (e.g. "net sales" vs. "revenue") —
paraphrasing and fusing results recovers those misses without changing the
embedding model.
Both are gated by `cfg.query.use_hyde` / `cfg.query.use_multi_query` so
ablation experiments can isolate their individual contribution to Hit Rate.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.query.models import TransformedQuery
from src.utils.llm_client import OllamaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

_HYDE_SYSTEM_PROMPT = """You write a short, plausible passage from a company's SEC 10-K or \
10-Q filing that would directly answer the given question. Write only the passage — no \
preamble, no "Here is...", no explanation. Use concrete numbers and financial terminology \
even if invented, since this passage is only used to find real matching text, never shown \
to a user."""

_HYDE_USER_TEMPLATE = "Question: {query}\nPassage:"

_MULTI_QUERY_SYSTEM_PROMPT = """You rewrite a financial question into {n} alternative \
phrasings that ask the same thing using different wording (e.g. synonyms for financial \
terms, different sentence structure). Respond with exactly {n} lines, one rewrite per \
line, no numbering, no extra commentary."""

_MULTI_QUERY_USER_TEMPLATE = "Question: {query}\nRewrites:"


class QueryTransformer:
    """Wraps the LLM calls behind HyDE document generation and multi-query paraphrasing."""

    def __init__(self, cfg: DictConfig, llm: OllamaClient | None = None) -> None:
        """Build the transformer; reuse injected `llm` in tests, else build from cfg.generation."""
        self.cfg = cfg
        self.llm = llm or OllamaClient(model=cfg.generation.model, host=cfg.generation.ollama_host)

    def generate_hyde(self, query: str) -> str:
        """Generate one hypothetical filing passage that would answer `query`."""
        return self.llm.complete(
            _HYDE_USER_TEMPLATE.format(query=query),
            system=_HYDE_SYSTEM_PROMPT,
            temperature=0.3,
        ).strip()

    def generate_multi_queries(self, query: str, n: int | None = None) -> list[str]:
        """Generate up to `n` paraphrases of `query` (defaults to cfg.query.num_multi_queries).

        WHY filter out lines matching `query` itself: local models sometimes
        echo the original question as one of the "rewrites" despite the
        prompt asking for alternative phrasings — an exact duplicate adds no
        retrieval coverage and wastes a retrieval call downstream.
        """
        n = n if n is not None else self.cfg.query.num_multi_queries
        raw_response = self.llm.complete(
            _MULTI_QUERY_USER_TEMPLATE.format(query=query, n=n),
            system=_MULTI_QUERY_SYSTEM_PROMPT.format(n=n),
            temperature=0.5,
        )
        candidates = [line.strip(" -\t") for line in raw_response.splitlines()]
        rewrites = [line for line in candidates if line and line.lower() != query.strip().lower()]
        return rewrites[:n]

    def transform(self, query: str) -> TransformedQuery:
        """Run whichever of HyDE / multi-query are enabled in `cfg.query` and bundle the results."""
        hyde_doc = self.generate_hyde(query) if self.cfg.query.use_hyde else None
        multi_queries = self.generate_multi_queries(query) if self.cfg.query.use_multi_query else []
        logger.debug(
            "Transformed query=%r: hyde=%s multi_queries=%d",
            query,
            hyde_doc is not None,
            len(multi_queries),
        )
        return TransformedQuery(raw_query=query, hyde_doc=hyde_doc, multi_queries=multi_queries)
