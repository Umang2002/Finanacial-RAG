"""Dense embedding model wrapper: BGE-M3 via sentence-transformers (free, local).

WHAT: BGEEmbedder loads BAAI/bge-m3 once and exposes a batched embed() that
returns L2-normalized vectors, ready for cosine-similarity search in Qdrant.
WHY a thin wrapper instead of calling SentenceTransformer directly everywhere:
keeps the model load (slow, ~2.3GB download on first run) in one place, and
gives dense_indexer.py / index_manager.py a single seam to mock in tests.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """Loads BAAI/bge-m3 once and embeds text batches into 1024-dim dense vectors."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32) -> None:
        """Load the sentence-transformers model onto the best available device.

        Args:
            model_name: HuggingFace model id, from config embeddings.dense_model.
            batch_size: encode batch size, from config embeddings.batch_size —
                kept low relative to OpenAI-API batch sizes since this runs
                on local CPU/GPU memory, not a remote service.
        """
        logger.info(f"Loading dense embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into normalized dense vectors.

        Args:
            texts: chunk texts to embed.

        Returns:
            np.ndarray of shape (len(texts), 1024).

        WHY normalize_embeddings=True: collection uses cosine distance —
        Qdrant's cosine metric expects (or at least is consistent with)
        unit-normalized vectors, and BGE-M3's own usage guide normalizes
        before similarity scoring.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
