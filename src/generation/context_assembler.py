"""Packs retrieved chunks into the context window; handles lost-in-middle reordering.

WHAT: ContextAssembler.assemble() turns Phase 5's ranked RetrievedChunks
into one prompt-ready text block, each chunk tagged with a [n] citation
number, plus the Citation list needed to resolve those numbers later.
WHY reorder chunks before assembling text: LLMs attend less to content
placed in the middle of a long context ("lost in the middle"). Citation
numbers stay in relevance order (1 = most relevant) so the model still
sees rank in the numbering, but the most relevant chunks are physically
placed at the start AND end of the context block, least relevant in the
middle.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.generation.models import AssembledContext, Citation
from src.retrieval.models import RetrievedChunk


def _sandwich_reorder(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder chunks so the most relevant land at both ends, least relevant in the middle.

    Input is assumed sorted best-first (Phase 5's reranked output). Walks
    that order, alternately filling from the front and back of the result
    list, e.g. [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2].
    """
    n = len(chunks)
    result: list[RetrievedChunk | None] = [None] * n
    left, right = 0, n - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return result  # type: ignore[return-value]


class ContextAssembler:
    """Builds the citation-tagged context block handed to the Phase 6 LLM prompt."""

    def __init__(self, cfg: DictConfig) -> None:
        """No state beyond config — kept as a class for symmetry with the rest of Phase 5/6."""
        self.cfg = cfg

    def assemble(self, chunks: list[RetrievedChunk]) -> AssembledContext:
        """Assign citation numbers in relevance order, then place chunk text in sandwich order.

        Args:
            chunks: reranked candidates from RetrievalPipeline.retrieve(),
                best-first.
        """
        citations = [_to_citation(i + 1, chunk) for i, chunk in enumerate(chunks)]
        citation_by_chunk_id = {c.chunk_id: c for c in citations}

        ordered_chunks = _sandwich_reorder(chunks)
        blocks = []
        for chunk in ordered_chunks:
            citation = citation_by_chunk_id[chunk.chunk.chunk_id]
            blocks.append(_format_block(citation, chunk))

        return AssembledContext(text="\n\n".join(blocks), citations=citations)


def _to_citation(citation_id: int, chunk: RetrievedChunk) -> Citation:
    """Build a Citation from a RetrievedChunk's metadata, defaulting unknown fields to empty."""
    meta = chunk.chunk.metadata
    return Citation(
        citation_id=citation_id,
        chunk_id=chunk.chunk.chunk_id,
        ticker=meta.ticker if meta else "",
        form=meta.form if meta else "",
        fiscal_year=meta.fiscal_year if meta else 0,
        item_label=meta.item_label if meta else "",
    )


def _format_block(citation: Citation, chunk: RetrievedChunk) -> str:
    """Format one context block: [n] (identity) followed by the chunk's text."""
    identity = f"{citation.ticker} {citation.form} FY{citation.fiscal_year}, {citation.item_label}"
    text = chunk.chunk.context_text or chunk.chunk.text
    return f"[{citation.citation_id}] ({identity})\n{text}"
