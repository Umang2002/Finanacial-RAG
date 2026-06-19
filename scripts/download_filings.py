"""CLI: download SEC 10-K/10-Q filings into data/raw/ for given tickers/years.

Usage:
    python scripts/download_filings.py --ticker AAPL --years 2022 2023
    python scripts/download_filings.py  # uses configs/base.yaml defaults

WHY a thin CLI: all real logic lives in SECEdgarLoader
(src/ingestion/sec_loader.py) so it stays testable without argparse/sys.argv
plumbing. This script only wires config -> loader -> summary table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# LEARN: allow `python scripts/download_filings.py` from the repo root
# without installing the package — add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.ingestion.sec_loader import SECEdgarLoader
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for tickers/years/forms/config path.

    WHY: all flags default to None so config values pass through unchanged
    when a flag isn't given — only an explicit --ticker/--years/--forms
    narrows the run.
    """
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
        help="Fiscal years to fetch (default: config ingestion.years)",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        choices=["10-K", "10-Q"],
        help="Filing types (default: config ingestion.filing_types)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: configs/base.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, run SECEdgarLoader.download_all, print a rich summary table."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    tickers = args.tickers or list(cfg.ingestion.tickers)
    years = args.years or list(cfg.ingestion.years)
    forms = args.forms or list(cfg.ingestion.filing_types)

    loader = SECEdgarLoader(
        user_agent=cfg.ingestion.sec_user_agent,
        raw_data_dir=cfg.ingestion.raw_data_dir,
    )

    logger.info(f"Downloading {forms} for {tickers} x {years}")
    results = loader.download_all(tickers, forms, years)

    table = Table(title="Downloaded Filings")
    table.add_column("Ticker")
    table.add_column("Form")
    table.add_column("Fiscal Year")
    table.add_column("Size")
    table.add_column("Status")
    for filing in results:
        table.add_row(
            filing.ticker,
            filing.form,
            str(filing.fiscal_year),
            f"{filing.size_bytes:,} B",
            "skipped (cached)" if filing.skipped else "downloaded",
        )

    Console().print(table)
    new_count = sum(1 for f in results if not f.skipped)
    logger.info(f"Done — {len(results)} filings ({new_count} new)")


if __name__ == "__main__":
    main()
