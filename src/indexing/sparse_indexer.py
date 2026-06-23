"""BM25 sparse vector encoding for Qdrant's native sparse-vector support.

WHAT: BM25SparseEncoder builds a corpus-wide vocabulary + IDF table with
rank-bm25, then converts each document into a Qdrant SparseVector (token-id ->
BM25 weight). The same vocab+idf is persisted so query-time encoding (Phase 5
retrieval) stays consistent with what was indexed.
WHY not rank-bm25's own .get_scores(): that does a full corpus scan per query
at search time — fine for in-memory toy search, but we want Qdrant's sparse
HNSW index to do the matching, so each document's BM25 weights must be
materialized as a vector once, up front.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qdrant_client import models
from rank_bm25 import BM25Okapi

from src.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization — shared by indexing and query-time encoding.

    WHY regex instead of a real tokenizer library: BM25 only needs stable,
    cheap term splitting; pulling in nltk/spacy for this would be a new
    dependency for no retrieval-quality benefit at this corpus size.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25SparseEncoder:
    """Fits BM25 over a corpus and encodes documents/queries as Qdrant SparseVectors."""

    def __init__(self) -> None:
        """Start unfit — call fit() with the full corpus before encode_all()/encode_query()."""
        self._bm25: BM25Okapi | None = None
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        """Tokenize the full corpus, fit BM25Okapi, and assign each unique term a stable int id.

        WHY fit on the full corpus, not per-filing: BM25 IDF is only
        meaningful relative to the whole document collection — fitting per
        filing would make every term's idf identical within that filing
        (no discriminative power) and incompatible across filings sharing
        one Qdrant collection.
        """
        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._vocab = {}
        for doc_tokens in tokenized:
            for term in doc_tokens:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self._idf = self._bm25.idf
        logger.info(f"BM25 fit: {len(texts)} docs, vocab size {len(self._vocab)}")

    def encode_all(self) -> list[models.SparseVector]:
        """Encode every document the encoder was fit on into a Qdrant SparseVector.

        Returns:
            One SparseVector per document, in the same order passed to fit().

        WHY reuse bm25.doc_freqs/idf/doc_len/avgdl instead of recomputing:
        rank-bm25 already computed per-document term frequencies during
        fit() — recomputing the BM25 weight formula here just turns that
        internal state into the (term_id, weight) pairs Qdrant needs, with
        no second pass over the raw text.
        """
        assert self._bm25 is not None, "call fit() before encode_all()"
        bm25 = self._bm25
        vectors: list[models.SparseVector] = []
        for i, term_freqs in enumerate(bm25.doc_freqs):
            doc_len = bm25.doc_len[i]
            indices: list[int] = []
            values: list[float] = []
            for term, freq in term_freqs.items():
                idf = bm25.idf.get(term, 0.0)
                if idf == 0.0:
                    continue
                denom = freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
                weight = idf * freq * (bm25.k1 + 1) / denom
                indices.append(self._vocab[term])
                values.append(float(weight))
            vectors.append(models.SparseVector(indices=indices, values=values))
        return vectors

    def encode_query(self, query: str) -> models.SparseVector:
        """Encode a query into a SparseVector using IDF-weighted term presence.

        WHY no BM25 length-normalization on the query side: Qdrant scores a
        sparse query against indexed documents via dot product — the
        document vectors already carry the full BM25 weight (including
        length normalization), so the query vector only needs to select
        which terms matter and by how much they matter (IDF), matching the
        convention used by Qdrant's own bm25 fastembed model.

        WHY self._idf not self._bm25.idf: at query time (Phase 5 retrieval)
        there's no fitted BM25Okapi instance — load_vocab() populates
        self._idf directly from the persisted file without needing the
        full corpus in memory again.
        """
        assert self._vocab, "call fit() or load_vocab() before encode_query()"
        term_freqs: dict[str, int] = {}
        for term in tokenize(query):
            if term in self._vocab:
                term_freqs[term] = term_freqs.get(term, 0) + 1
        indices = [self._vocab[t] for t in term_freqs]
        values = [float(self._idf.get(t, 0.0) * f) for t, f in term_freqs.items()]
        return models.SparseVector(indices=indices, values=values)

    def load_vocab(self, path: str | Path) -> None:
        """Load a persisted vocab+idf (written by save_vocab) for query-time encoding only.

        WHY this doesn't restore a full BM25Okapi: encode_query() only ever
        reads idf + the vocab's term->id mapping, never doc_freqs/doc_len —
        so query-time retrieval needs none of the per-document state that
        fit() builds, just the term statistics from the indexed corpus.
        """
        data = json.loads(Path(path).read_text())
        self._vocab = {k: int(v) for k, v in data["vocab"].items()}
        self._idf = data["idf"]
        logger.info(f"BM25 vocab+idf loaded from {path} ({len(self._vocab)} terms)")

    def save_vocab(self, path: str | Path) -> None:
        """Persist vocab + idf so retrieval (Phase 5) can encode queries with the same mapping.

        WHY persist rather than refit at query time: refitting BM25 at
        query time would require the full indexed corpus in memory again,
        and idf would silently drift if the corpus changes between indexing
        and serving.
        """
        assert self._bm25 is not None, "call fit() before save_vocab()"
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "vocab": self._vocab,
                    "idf": self._bm25.idf,
                    "k1": self._bm25.k1,
                    "b": self._bm25.b,
                    "avgdl": self._bm25.avgdl,
                }
            )
        )
        logger.info(f"BM25 vocab+idf saved to {out_path}")
