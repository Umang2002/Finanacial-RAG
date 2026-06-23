"""CLI: run the full Phase 1-6 query pipeline end to end for one question.

Usage:
    python scripts/generate.py --query "What was Apple's FY2023 net sales?"

WHY a thin CLI: real logic lives in src/query/*.py, src/retrieval/*.py, and
src/generation/*.py so it stays testable without argparse/sys.argv
plumbing. This script only wires config -> intent classification -> HyDE/
multi-query expansion -> decomposition (if multi-hop) -> retrieval ->
generation, and prints the final cited answer (same pattern as
scripts/retrieve.py, which stops before Phase 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.generation.generator import Generator
from src.query.query_analyzer import QueryAnalyzer
from src.query.query_decomposer import QueryDecomposer
from src.query.query_transformer import QueryTransformer
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse the query string + optional config override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument(
        "--config", default=None, help="Path to config YAML (default: configs/base.yaml)"
    )
    return parser.parse_args()


def main() -> None:
    """Load config, run the full query -> retrieve -> generate pipeline, print the cited answer."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    analyzer = QueryAnalyzer(cfg)
    transformer = QueryTransformer(cfg)
    decomposer = QueryDecomposer(cfg)
    retrieval_pipeline = RetrievalPipeline(cfg)
    generator = Generator(cfg)

    analyzed = analyzer.analyze(args.query)
    logger.info(f"Intent={analyzed.intent} multi_hop={analyzed.is_multi_hop}")

    transformed = transformer.transform(args.query)
    decomposed = decomposer.decompose(args.query) if analyzed.is_multi_hop else None

    chunks = retrieval_pipeline.retrieve(args.query, transformed=transformed, decomposed=decomposed)
    result = generator.generate(args.query, chunks)

    console.print(f"\n[bold]Question:[/bold] {result.query}")
    console.print(f"\n[bold]Answer:[/bold] {result.answer}")
    if result.confidence is not None:
        console.print(f"[bold]Confidence:[/bold] {result.confidence:.2f}")

    table = Table(title="Citations")
    table.add_column("ID")
    table.add_column("Ticker/Form/Year")
    table.add_column("Section")
    for citation in result.citations:
        identity = f"{citation.ticker}/{citation.form}/{citation.fiscal_year}"
        table.add_row(f"[{citation.citation_id}]", identity, citation.item_label)
    console.print(table)


if __name__ == "__main__":
    main()
