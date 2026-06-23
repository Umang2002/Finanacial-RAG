"""Unit tests for BM25SparseEncoder (src/indexing/sparse_indexer.py).

WHY no mocking needed here: rank-bm25 is pure CPU math over small tokenized
lists — unlike BGE-M3, fitting it in a test is fast and exercises the real
weight formula, not a stand-in.
"""

from __future__ import annotations

import json

from qdrant_client import models

from src.indexing.sparse_indexer import BM25SparseEncoder, tokenize


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Apple's Revenue Grew 10%!") == ["apple", "s", "revenue", "grew", "10"]


def test_fit_assigns_stable_ids_to_every_unique_term() -> None:
    encoder = BM25SparseEncoder()
    encoder.fit(["apple revenue", "apple iphone sales"])
    assert set(encoder._vocab) == {"apple", "revenue", "iphone", "sales"}
    # ids are unique and contiguous
    assert sorted(encoder._vocab.values()) == [0, 1, 2, 3]


def test_encode_all_gives_higher_weight_to_rarer_terms() -> None:
    # "apple" appears in all 3 docs (low idf); "iphone" only in doc 1 (high idf)
    encoder = BM25SparseEncoder()
    encoder.fit(["apple revenue apple", "apple iphone sales", "apple stock price"])
    vectors = encoder.encode_all()

    assert len(vectors) == 3
    assert all(isinstance(v, models.SparseVector) for v in vectors)

    doc1_terms = dict(zip(vectors[1].indices, vectors[1].values))
    iphone_weight = doc1_terms[encoder._vocab["iphone"]]
    apple_weight = doc1_terms[encoder._vocab["apple"]]
    assert iphone_weight > apple_weight  # rarer term gets more weight


def test_encode_query_ignores_out_of_vocab_terms() -> None:
    encoder = BM25SparseEncoder()
    encoder.fit(["apple revenue", "iphone sales"])

    vec = encoder.encode_query("apple smartphone")  # "smartphone" never seen
    assert len(vec.indices) == 1
    assert vec.indices[0] == encoder._vocab["apple"]


def test_save_vocab_writes_vocab_idf_and_bm25_params(tmp_path) -> None:
    encoder = BM25SparseEncoder()
    encoder.fit(["apple revenue", "iphone sales"])
    out_path = tmp_path / "vocab.json"

    encoder.save_vocab(out_path)

    data = json.loads(out_path.read_text())
    assert set(data["vocab"]) == {"apple", "revenue", "iphone", "sales"}
    assert "idf" in data and "k1" in data and "b" in data and "avgdl" in data
