"""Parses structured LLM output: answer text, inline citations, and confidence score.

WHAT: parse_llm_output() takes the raw chat completion text (answer with
[n] markers and an optional trailing "Confidence: 0.8" line) and splits it
into clean answer text, the subset of available Citations actually
referenced, and the confidence score.
WHY only citations actually referenced are kept (not every chunk handed to
the LLM): GeneratedAnswer.citations should reflect what the answer is
grounded in — a chunk that was in context but never cited isn't evidence
for anything the model said.
"""

from __future__ import annotations

import re

from src.generation.models import Citation

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CONFIDENCE_PATTERN = re.compile(r"\n?confidence:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def parse_llm_output(
    raw_text: str, available_citations: list[Citation]
) -> tuple[str, list[Citation], float | None]:
    """Split raw LLM text into (clean answer, referenced citations, confidence).

    Args:
        raw_text: the LLM's full response, e.g. "Revenue was $383B [1].\\nConfidence: 0.9"
        available_citations: every Citation ContextAssembler defined for
            this prompt, used to resolve [n] markers to filing identity.
    """
    confidence_match = _CONFIDENCE_PATTERN.search(raw_text)
    confidence: float | None = None
    answer_text = raw_text
    if confidence_match:
        confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
        answer_text = raw_text[: confidence_match.start()]
    answer_text = answer_text.strip()

    citation_by_id = {c.citation_id: c for c in available_citations}
    seen_ids: set[int] = set()
    referenced: list[Citation] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        citation_id = int(match.group(1))
        if citation_id in seen_ids:
            continue
        citation = citation_by_id.get(citation_id)
        if citation is not None:
            seen_ids.add(citation_id)
            referenced.append(citation)

    return answer_text, referenced, confidence
