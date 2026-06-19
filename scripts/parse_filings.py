"""CLI: parse downloaded SEC filings (data/raw/) into clean structured JSON (data/processed/).

Usage:
    python scripts/parse_filings.py --ticker AAPL --years 2023
    python scripts/parse_filings.py  # uses configs/base.yaml defaults

WHY a thin CLI: real logic lives in HTMLFilingParser (src/ingestion/html_parser.py)
so it stays testable without argparse/sys.argv plumbing. This script only
wires config -> reads data/raw/{ticker}/{form}_{year}/metadata.json -> parser
-> data/processed/{ticker}/{form}_{year}/parsed.json -> summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.ingestion.html_parser import HTMLFilingParser
from src.ingestion.models import FilingRef
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for tickers/years/forms/config — same flags as download_filings.py."""
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
        help="Fiscal years to parse (default: config ingestion.years)",
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


def _find_raw_filings(
    raw_dir: Path, tickers: list[str], forms: list[str], years: list[int]
) -> list[tuple[FilingRef, Path]]:
    """Scan data/raw/{ticker}/{form}_{year}/ for metadata.json + primary doc pairs.

    WHY scan disk instead of re-listing from SEC: parsing is a separate,
    idempotent step from downloading — it should work offline against
    whatever's already on disk, matching the metadata.json each
    download_filing() call already wrote.
    """
    found: list[tuple[FilingRef, Path]] = []
    for ticker in tickers:
        for form in forms:
            for year in years:
                filing_dir = raw_dir / ticker / f"{form}_{year}"
                metadata_path = filing_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                # DownloadedFiling JSON has more fields than FilingRef needs —
                # parse the dict directly rather than round-tripping a model.
                raw_meta = json.loads(metadata_path.read_text())
                ref = FilingRef(
                    ticker=raw_meta["ticker"],
                    cik=raw_meta["cik"],
                    form=raw_meta["form"],
                    filing_date=raw_meta["filing_date"],
                    report_date=raw_meta["report_date"],
                    accession_no=raw_meta["accession_no"],
                    primary_doc=raw_meta["primary_doc"],
                )
                primary_path = next(filing_dir.glob("primary.*"), None)
                if primary_path is None:
                    logger.warning(
                        f"{filing_dir}: metadata.json found but no primary.* file — skipping"
                    )
                    continue
                found.append((ref, primary_path))
    return found


def main() -> None:
    """Load config, parse every matching raw filing on disk, write parsed.json, print summary."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    tickers = args.tickers or list(cfg.ingestion.tickers)
    years = args.years or list(cfg.ingestion.years)
    forms = args.forms or list(cfg.ingestion.filing_types)

    raw_dir = Path(cfg.ingestion.raw_data_dir)
    processed_dir = Path(cfg.ingestion.processed_data_dir)

    pairs = _find_raw_filings(raw_dir, tickers, forms, years)
    logger.info(f"Found {len(pairs)} raw filings on disk matching {forms} x {tickers} x {years}")

    parser_ = HTMLFilingParser()
    table = Table(title="Parsed Filings")
    table.add_column("Ticker")
    table.add_column("Form")
    table.add_column("Fiscal Year")
    table.add_column("Sections")
    table.add_column("Tables")
    table.add_column("Status")

    for ref, primary_path in pairs:
        out_dir = processed_dir / ref.ticker / f"{ref.form}_{ref.fiscal_year}"
        out_path = out_dir / "parsed.json"
        if out_path.exists():
            table.add_row(ref.ticker, ref.form, str(ref.fiscal_year), "-", "-", "skipped (cached)")
            continue

        parsed = parser_.parse(primary_path, ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(parsed.model_dump_json(indent=2))
        table.add_row(
            ref.ticker,
            ref.form,
            str(ref.fiscal_year),
            str(len(parsed.sections)),
            str(len(parsed.raw_tables)),
            "parsed",
        )

    Console().print(table)


if __name__ == "__main__":
    main()
