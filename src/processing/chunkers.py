"""All chunking strategies: fixed, recursive, semantic, sentence-window, parent-child.

WHAT: each chunk_*() function takes cleaned section/table text and returns
list[Chunk] (no metadata yet — enricher.py stamps filing identity on after).
chunk_text() dispatches to the configured strategy via cfg.chunking.strategy.
WHY one dispatch function instead of a class hierarchy: every strategy is a
pure function of (text, config) -> list[Chunk] — a registry dict is simpler
than a Strategy-pattern class for five functions with no shared state, per
CLAUDE.md's no-speculative-abstraction rule.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from src.processing.models import Chunk, ChunkStrategy
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)

# LEARN: splits after sentence-ending punctuation followed by whitespace and
# a capital letter/quote/paren — a simple heuristic (not a full NLP sentence
# tokenizer) but sufficient for 10-K/10-Q prose, and avoids adding an nltk/
# spacy dependency for a feature only sentence_window/semantic need.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

# LEARN: lazy module-level singleton — sentence-transformers' BGE-M3 load is
# a ~2.3GB download + multi-second model init. Only chunk_semantic() touches
# this; fixed/recursive/sentence_window/parent_child users never pay that
# cost.
_embedder = None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns [] for empty input."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _make_chunk(
    text: str,
    strategy: ChunkStrategy,
    chunk_index: int,
    *,
    parent_id: str | None = None,
    is_parent: bool = False,
    context_text: str | None = None,
) -> Chunk:
    """Build a Chunk with a fresh chunk_id and a word-count token_count proxy.

    WHY word count, not a real tokenizer: no tokenizer dependency (tiktoken/
    transformers AutoTokenizer) is wired in at this phase — Phase 3's
    embedding model does real tokenization at encode time. This is only
    used for chunk-size sanity-checking/logging, not for hitting an exact
    token budget.
    """
    return Chunk(
        chunk_id=str(uuid4()),
        text=text,
        strategy=strategy,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        parent_id=parent_id,
        is_parent=is_parent,
        context_text=context_text,
    )


def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split text into fixed-size character windows with overlap. Simplest baseline strategy."""
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_size - chunk_overlap, 1)
    idx = 0
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece.strip():
            chunks.append(_make_chunk(piece, "fixed", idx))
            idx += 1
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_recursive(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str]
) -> list[Chunk]:
    """Split text hierarchically on configured separators (paragraph -> line -> sentence -> word).

    WHY langchain's splitter instead of hand-rolled: RecursiveCharacterTextSplitter
    already implements "try each separator in order, recurse into oversized
    pieces" correctly including overlap bookkeeping — reimplementing that is
    pure risk for zero benefit (already a project dependency, runs fully local).
    """
    if not text:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=list(separators)
    )
    pieces = splitter.split_text(text)
    return [_make_chunk(piece, "recursive", i) for i, piece in enumerate(pieces)]


def chunk_sentence_window(text: str, window_size: int) -> list[Chunk]:
    """One chunk per sentence (precise embedding target); context_text holds the surrounding window.

    WHY: embedding a single sentence gets a tight semantic match for a
    specific fact, but a single sentence is often too little context for the
    LLM to answer from — context_text (±window_size sentences) is what
    generation actually reads, while `text` is what gets indexed/searched.
    """
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    for i, sentence in enumerate(sentences):
        lo, hi = max(0, i - window_size), min(len(sentences), i + window_size + 1)
        context = " ".join(sentences[lo:hi])
        chunks.append(_make_chunk(sentence, "sentence_window", i, context_text=context))
    return chunks


def chunk_parent_child(
    text: str, parent_chunk_size: int, child_chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split into large parent chunks, then each parent into small children pointing back at it.

    WHY: small chunks embed precisely (good retrieval match) but lack
    context; large chunks have context but dilute the embedding (bad
    retrieval match). Indexing only the children and fetching the parent at
    generation time gets both — this is why is_parent=True chunks should be
    excluded from the vector index in Phase 3 and looked up by parent_id
    instead.
    """
    if not text:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Chunk] = []
    child_idx = 0
    for parent_idx, parent_text in enumerate(parent_splitter.split_text(text)):
        parent_chunk = _make_chunk(parent_text, "parent_child", parent_idx, is_parent=True)
        chunks.append(parent_chunk)
        for child_text in child_splitter.split_text(parent_text):
            chunks.append(
                _make_chunk(child_text, "parent_child", child_idx, parent_id=parent_chunk.chunk_id)
            )
            child_idx += 1
    return chunks


def _get_embedder(model_name: str):
    """Lazily load + cache the sentence-transformers model used for semantic chunking."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model for semantic chunking: {model_name}")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def chunk_semantic(
    text: str, model_name: str, similarity_threshold: float, max_chunk_size: int
) -> list[Chunk]:
    """Group consecutive sentences into a chunk while cosine similarity stays above threshold.

    WHY embedding-based breakpoints instead of fixed size: a chunk boundary
    placed mid-topic hurts retrieval (half the relevant context lands in the
    neighboring chunk). Walking sentence-to-sentence similarity and cutting
    where it drops below threshold approximates topic boundaries instead of
    cutting at an arbitrary character count.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [_make_chunk(sentences[0], "semantic", 0)]

    embedder = _get_embedder(model_name)
    embeddings = embedder.encode(sentences, normalize_embeddings=True)

    groups: list[list[str]] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
        current_len = sum(len(s) for s in current)
        if similarity < similarity_threshold or current_len >= max_chunk_size:
            groups.append(current)
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    groups.append(current)

    return [_make_chunk(" ".join(group), "semantic", i) for i, group in enumerate(groups)]


def chunk_text(text: str, cfg: DictConfig) -> list[Chunk]:
    """Dispatch to the chunking strategy configured in cfg.chunking.strategy.

    WHY taking the whole cfg instead of unpacking args at the call site:
    each strategy needs a different subset of chunking.* keys — centralizing
    the unpacking here means scripts/process_filings.py doesn't need to know
    which keys matter for which strategy.
    """
    strategy = cfg.chunking.strategy
    if strategy == "fixed":
        return chunk_fixed(text, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    if strategy == "recursive":
        return chunk_recursive(
            text, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap, cfg.chunking.separators
        )
    if strategy == "sentence_window":
        return chunk_sentence_window(text, cfg.chunking.sentence_window_size)
    if strategy == "parent_child":
        return chunk_parent_child(
            text,
            cfg.chunking.parent_chunk_size,
            cfg.chunking.child_chunk_size,
            cfg.chunking.chunk_overlap,
        )
    if strategy == "semantic":
        return chunk_semantic(
            text,
            cfg.embeddings.dense_model,
            cfg.chunking.semantic_similarity_threshold,
            cfg.chunking.chunk_size,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
