"""Unit tests for BGEEmbedder (src/indexing/embeddings.py).

WHY monkeypatch SentenceTransformer instead of loading BAAI/bge-m3: same
reasoning as the semantic chunker test (tests/unit/test_chunkers.py) —
loading the real model is a ~2.3GB download + multi-second init. Patching
the class lets embed() exercise the real batching/normalize_embeddings
wiring against a fake that returns known vectors.
"""

from __future__ import annotations

import numpy as np

from src.indexing import embeddings as embeddings_module
from src.indexing.embeddings import BGEEmbedder


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append({"texts": texts, **kwargs})
        return np.ones((len(texts), 4), dtype=float)


def test_embed_passes_batch_size_and_normalize_flag(monkeypatch) -> None:
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = BGEEmbedder(model_name="fake-model", batch_size=16)

    result = embedder.embed(["a", "b", "c"])

    assert result.shape == (3, 4)
    call = embedder.model.encode_calls[0]
    assert call["batch_size"] == 16
    assert call["normalize_embeddings"] is True


def test_embed_returns_numpy_array(monkeypatch) -> None:
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = BGEEmbedder(model_name="fake-model")

    result = embedder.embed(["hello"])

    assert isinstance(result, np.ndarray)
