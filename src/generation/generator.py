"""LLM call with citation-aware system prompt; returns answer grounded in retrieved context.

WHAT: Generator.generate() assembles context from Phase 5's reranked
chunks, prompts the local LLM to answer using only that context with
inline [n] citations, and parses the response into a GeneratedAnswer.
WHY citation_style only supports "inline": configs/base.yaml declares
footnote/list as future options, but no prompt/parser for them exists yet
— failing fast on an unsupported value beats silently producing
unparsable output.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.generation.context_assembler import ContextAssembler
from src.generation.models import GeneratedAnswer
from src.generation.output_parser import parse_llm_output
from src.retrieval.models import RetrievedChunk
from src.utils.llm_client import OllamaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You answer questions about SEC financial filings (10-K/10-Q) using ONLY \
the numbered context passages provided. Every factual claim must cite the passage it came \
from using its bracket number, e.g. [1]. If the context does not contain enough information \
to answer, say so plainly instead of guessing. Do not use outside knowledge.

After the answer, on its own line, report your confidence in the answer as a number between \
0 and 1, formatted exactly as: Confidence: 0.8"""

_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}
Answer:"""


class Generator:
    """Wires ContextAssembler + the local LLM + output_parser into one generate() call."""

    def __init__(self, cfg: DictConfig, llm: OllamaClient | None = None) -> None:
        """Build the generator; reuse injected `llm` in tests, else build from cfg.generation."""
        if cfg.generation.citation_style != "inline":
            raise ValueError(
                f"Unsupported citation_style={cfg.generation.citation_style!r}; "
                "only 'inline' is implemented"
            )
        self.cfg = cfg
        self.llm = llm or OllamaClient(model=cfg.generation.model, host=cfg.generation.ollama_host)
        self.assembler = ContextAssembler(cfg)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Assemble context from `chunks`, ask the LLM to answer `query`, and parse the result."""
        assembled = self.assembler.assemble(chunks)
        raw_response = self.llm.complete(
            _USER_PROMPT_TEMPLATE.format(context=assembled.text, query=query),
            system=_SYSTEM_PROMPT,
            temperature=self.cfg.generation.temperature,
        )
        answer_text, citations, confidence = parse_llm_output(raw_response, assembled.citations)
        logger.debug(
            "Generated answer for query=%r with %d citation(s), confidence=%s",
            query,
            len(citations),
            confidence,
        )
        return GeneratedAnswer(
            query=query, answer=answer_text, citations=citations, confidence=confidence
        )
