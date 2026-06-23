"""CLI: test the retrieval pipeline interactively against one query.

Usage:
    python scripts/retrieve.py --query "What was Apple's FY2023 net sales?"

WHY a thin CLI: real logic lives in src/retrieval/*.py so it stays testable
without argparse/sys.argv plumbing. This script only wires config -> runs
dense, sparse, hybrid (RRF), and reranked retrieval against the real Qdrant
collection, and prints each stage so a single query's behavior can be
inspected end to end (same pattern as scripts/build_index.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import rrf_fuse
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_retriever import SparseRetriever
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse the query string + optional config override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Query to retrieve for")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Rows to display per stage (default 5)"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML (default: configs/base.yaml)"
    )
    return parser.parse_args()


def _print_hits(title: str, hits: list[RetrievedChunk], top_k: int) -> None:
    """Print one stage's top_k hits as a rich table: score, filing identity, text preview."""
    table = Table(title=title)
    table.add_column("Score")
    table.add_column("Ticker/Form/Year")
    table.add_column("Section")
    table.add_column("Text Preview")

    for hit in hits[:top_k]:
        meta = hit.chunk.metadata
        identity = f"{meta.ticker}/{meta.form}/{meta.fiscal_year}" if meta else "—"
        section = meta.item_label if meta else "—"
        preview = hit.chunk.text[:80].replace("\n", " ") + "..."
        table.add_row(f"{hit.score:.4f}", identity, section, preview)

    console.print(table)


def _print_comparison(dense_hits: list[RetrievedChunk], sparse_hits: list[RetrievedChunk]) -> None:
    """Print which chunk ids appeared in dense-only, sparse-only, or both lists."""
    dense_ids = {hit.chunk.chunk_id for hit in dense_hits}
    sparse_ids = {hit.chunk.chunk_id for hit in sparse_hits}

    table = Table(title="Dense vs. Sparse Overlap")
    table.add_column("Dense-only")
    table.add_column("Sparse-only")
    table.add_column("Both")
    table.add_row(
        str(len(dense_ids - sparse_ids)),
        str(len(sparse_ids - dense_ids)),
        str(len(dense_ids & sparse_ids)),
    )
    console.print(table)


def main() -> None:
    """Load config, run dense/sparse/hybrid/reranked retrieval for one query, print each stage."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    client = QdrantClient(path=cfg.indexing.qdrant_path)
    dense_retriever = DenseRetriever(cfg, client)
    sparse_retriever = SparseRetriever(cfg, client)
    reranker = Reranker(cfg)

    dense_hits = dense_retriever.search(args.query)
    _print_hits("Dense Retrieval", dense_hits, args.top_k)

    sparse_hits = sparse_retriever.search(args.query)
    _print_hits("Sparse (BM25) Retrieval", sparse_hits, args.top_k)

    fused_hits = rrf_fuse([dense_hits, sparse_hits])
    _print_hits("Hybrid (RRF Fused) Retrieval", fused_hits, args.top_k)

    _print_comparison(dense_hits, sparse_hits)

    reranked_hits = reranker.rerank(args.query, fused_hits)
    _print_hits("Reranked (BGE Cross-Encoder)", reranked_hits, args.top_k)

    logger.info("Generation (Phase 6) not implemented yet — skipping final answer + citations")


if __name__ == "__main__":
    main()
