"""Pydantic models for query processing — output of Phase 4, input to Phase 5 retrieval.

WHAT: AnalyzedQuery carries intent classification; TransformedQuery carries
HyDE/multi-query expansions; DecomposedQuery carries multi-hop sub-questions.
WHY pydantic: these cross the query/*.py -> retrieval/*.py module boundary,
per CLAUDE.md "use pydantic for data structures that cross module
boundaries".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QueryIntent = Literal[
    "factual_lookup", "comparison", "calculation", "multi_hop", "definition", "other"
]


class AnalyzedQuery(BaseModel):
    """Intent classification result for one raw query."""

    raw_query: str
    intent: QueryIntent
    is_multi_hop: bool


class TransformedQuery(BaseModel):
    """HyDE hypothetical document + multi-query paraphrases for one raw query.

    WHY hyde_doc is optional but multi_queries defaults to empty list: HyDE
    is a single value that's either generated or not (None = skipped),
    multi-query is a set that's naturally empty when skipped — matching
    each field's shape avoids a sentinel value for "no expansion ran".
    """

    raw_query: str
    hyde_doc: str | None = None
    multi_queries: list[str] = Field(default_factory=list)


class DecomposedQuery(BaseModel):
    """Sub-questions for one query — single-element list when decomposition is off or skipped."""

    raw_query: str
    sub_questions: list[str]
