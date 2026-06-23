"""CLI: build the hybrid (dense + sparse) Qdrant index from chunks.json files.

Usage:
    python scripts/build_index.py --ticker AAPL --years 2023
    python scripts/build_index.py  # uses configs/base.yaml defaults

WHY a thin CLI: real logic lives in IndexManager (src/indexing/index_manager.py)
so it stays testable without argparse/sys.argv plumbing. This script only
wires config -> reads every data/processed/{ticker}/{form}_{year}/chunks.json
on disk -> IndexManager.build_index -> rich summary table, same pattern as
process_filings.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.indexing.index_manager import IndexManager
from src.processing.models import Chunk
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for tickers/years/forms/config — same flags as process_filings.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker symbol, repeatable (default: config ingestion.tickers)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Fiscal years to index (default: config ingestion.years)",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        choices=["10-K", "10-Q"],
        help="Filing types (default: config ingestion.filing_types)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML (default: configs/base.yaml)"
    )
    return parser.parse_args()


def _find_chunks(
    processed_dir: Path, tickers: list[str], forms: list[str], years: list[int]
) -> list[tuple[str, str, int, Path]]:
    """Scan data/processed/{ticker}/{form}_{year}/ for chunks.json files to index.

    WHY scan disk instead of re-running chunking: indexing is a separate,
    idempotent step from chunking — it should work offline against
    whatever process_filings.py already wrote, same pattern as
    process_filings.py reading parse_filings.py's output.
    """
    found: list[tuple[str, str, int, Path]] = []
    for ticker in tickers:
        for form in forms:
            for year in years:
                chunks_path = processed_dir / ticker / f"{form}_{year}" / "chunks.json"
                if chunks_path.exists():
                    found.append((ticker, form, year, chunks_path))
    return found


def main() -> None:
    """Load config, gather chunks from every matching filing, build the index, print a summary."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    tickers = args.tickers or list(cfg.ingestion.tickers)
    years = args.years or list(cfg.ingestion.years)
    forms = args.forms or list(cfg.ingestion.filing_types)

    processed_dir = Path(cfg.ingestion.processed_data_dir)
    found = _find_chunks(processed_dir, tickers, forms, years)
    logger.info(f"Found {len(found)} chunked filings matching {forms} x {tickers} x {years}")

    table = Table(title="Filings Indexed")
    table.add_column("Ticker")
    table.add_column("Form")
    table.add_column("Fiscal Year")
    table.add_column("Chunks")

    all_chunks: list[Chunk] = []
    for ticker, form, year, chunks_path in found:
        raw = json.loads(chunks_path.read_text())
        filing_chunks = [Chunk.model_validate(c) for c in raw]
        all_chunks.extend(filing_chunks)
        table.add_row(ticker, form, str(year), str(len(filing_chunks)))

    if not all_chunks:
        logger.warning("No chunks found — run scripts/process_filings.py first")
        return

    manager = IndexManager(cfg)
    n_points = manager.build_index(all_chunks)

    Console().print(table)
    logger.info(
        f"Indexed {n_points} chunks into collection '{cfg.indexing.collection_name}' "
        f"at {cfg.indexing.qdrant_path}"
    )


if __name__ == "__main__":
    main()
