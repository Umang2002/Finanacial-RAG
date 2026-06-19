"""Unit tests for all chunking strategies in src/processing/chunkers.py.

WHY chunk_semantic is tested with a fake embedder instead of BGE-M3: loading
the real model is a ~2.3GB download + multi-second init — monkeypatching
src.processing.chunkers._get_embedder keeps this test offline/fast while
still exercising the real grouping logic in chunk_semantic().
"""

from __future__ import annotations

import numpy as np
import pytest

from src.processing import chunkers
from src.processing.chunkers import (
    chunk_fixed,
    chunk_parent_child,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence_window,
    chunk_text,
)


class _FakeEmbedder:
    """Returns hand-picked unit vectors so cosine similarity between sentences is known exactly."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = np.array(vectors, dtype=float)

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        assert len(sentences) == len(self._vectors)
        return self._vectors


def test_chunk_fixed_respects_size_and_overlap() -> None:
    text = "a" * 100
    chunks = chunk_fixed(text, chunk_size=30, chunk_overlap=10)
    assert all(len(c.text) <= 30 for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)
    # last chunk reaches the end of the text
    assert "".join(chunks[0].text[:20]) == text[:20]


def test_chunk_fixed_empty_text_returns_nothing() -> None:
    assert chunk_fixed("", chunk_size=30, chunk_overlap=10) == []


def test_chunk_recursive_splits_on_paragraph_then_smaller_units() -> None:
    text = "First paragraph here.\n\nSecond paragraph here. It has two sentences."
    chunks = chunk_recursive(
        text, chunk_size=40, chunk_overlap=0, separators=["\n\n", "\n", ". ", " "]
    )
    assert len(chunks) >= 2
    assert all(c.strategy == "recursive" for c in chunks)
    assert all(
        len(c.text) <= 40 + 5 for c in chunks
    )  # langchain may slightly exceed at hard word boundaries


def test_chunk_sentence_window_anchors_one_sentence_with_context() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_sentence_window(text, window_size=1)
    assert [c.text for c in chunks] == [
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four.",
    ]
    # middle chunk's context_text includes its neighbors, not just itself
    assert chunks[1].context_text == "Sentence one. Sentence two. Sentence three."
    # first chunk has no left neighbor — window clamps instead of erroring
    assert chunks[0].context_text == "Sentence one. Sentence two."


def test_chunk_parent_child_links_children_to_parent_id() -> None:
    text = "Sentence one talks about revenue. " * 30
    chunks = chunk_parent_child(text, parent_chunk_size=200, child_chunk_size=50, chunk_overlap=0)

    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert parents
    assert children
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_chunk_semantic_breaks_on_low_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    # sentences 0,1 are near-identical (sim ~1.0); sentence 2 is orthogonal (sim 0.0)
    fake = _FakeEmbedder(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    monkeypatch.setattr(chunkers, "_get_embedder", lambda model_name: fake)

    text = "Revenue grew sharply. Revenue grew again. Risk factors include currency exposure."
    chunks = chunk_semantic(
        text, model_name="fake", similarity_threshold=0.5, max_chunk_size=10_000
    )

    assert len(chunks) == 2
    assert chunks[0].text == "Revenue grew sharply. Revenue grew again."
    assert chunks[1].text == "Risk factors include currency exposure."


def test_chunk_semantic_single_sentence_skips_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("embedder should not be loaded for a single sentence")

    monkeypatch.setattr(chunkers, "_get_embedder", _fail)
    chunks = chunk_semantic("Only one sentence here.", "fake", 0.5, 10_000)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."


class _Chunking:
    strategy = "fixed"
    chunk_size = 20
    chunk_overlap = 5
    separators = ["\n\n", "\n", ". ", " "]
    sentence_window_size = 1
    semantic_similarity_threshold = 0.5
    parent_chunk_size = 100
    child_chunk_size = 20


class _Embeddings:
    dense_model = "fake-model"


class _Cfg:
    """Minimal stand-in for the slice of DictConfig chunk_text() reads."""

    chunking = _Chunking
    embeddings = _Embeddings


def test_chunk_text_dispatches_to_configured_strategy() -> None:
    chunks = chunk_text("word " * 20, _Cfg)
    assert chunks
    assert all(c.strategy == "fixed" for c in chunks)


def test_chunk_text_unknown_strategy_raises() -> None:
    class _BadChunking(_Chunking):
        strategy = "bogus"

    class BadCfg(_Cfg):
        chunking = _BadChunking

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("some text", BadCfg)
