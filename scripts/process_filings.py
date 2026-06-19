"""CLI: chunk parsed SEC filings (data/processed/) into indexable Chunks (chunks.json).

Usage:
    python scripts/process_filings.py --ticker AAPL --years 2023
    python scripts/process_filings.py  # uses configs/base.yaml defaults

WHY a thin CLI: real logic lives in TextCleaner, TableSerializer, chunkers.py,
and enricher.py so it stays testable without argparse/sys.argv plumbing.
This script only wires config -> reads data/processed/{ticker}/{form}_{year}/parsed.json
-> clean -> chunk -> enrich -> data/processed/{ticker}/{form}_{year}/chunks.json -> summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.ingestion.models import ParsedFiling, ParsedSection
from src.processing.chunkers import chunk_text
from src.processing.cleaner import TextCleaner
from src.processing.enricher import enrich
from src.processing.models import Chunk
from src.processing.table_serializer import TableSerializer
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for tickers/years/forms/config path — same flags as parse_filings.py."""
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
        help="Fiscal years to process (default: config ingestion.years)",
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


def _find_parsed_filings(
    processed_dir: Path, tickers: list[str], forms: list[str], years: list[int]
) -> list[tuple[Path, Path]]:
    """Scan data/processed/{ticker}/{form}_{year}/ for parsed.json files to chunk.

    WHY scan disk instead of re-running the parser: chunking is a separate,
    idempotent step from parsing — it should work offline against whatever
    parse_filings.py already wrote, same pattern as parse_filings.py reading
    download_filings.py's output.
    """
    found: list[tuple[Path, Path]] = []
    for ticker in tickers:
        for form in forms:
            for year in years:
                filing_dir = processed_dir / ticker / f"{form}_{year}"
                parsed_path = filing_dir / "parsed.json"
                if parsed_path.exists():
                    found.append((parsed_path, filing_dir / "chunks.json"))
    return found


def chunk_filing(
    filing: ParsedFiling, cfg, cleaner: TextCleaner, serializer: TableSerializer
) -> list[Chunk]:
    """Clean + chunk + enrich every section, plus every serialized table, of one filing.

    WHY tables become synthetic sections: raw_tables is filing-level (not
    tied to a specific Item section) in ParsedFiling, but chunk_text() +
    enrich() only know how to operate on a ParsedSection — wrapping each
    serialized table as item_label="Table N" reuses that exact pipeline
    instead of a parallel table-only code path.
    """
    all_chunks: list[Chunk] = []

    for section in filing.sections:
        cleaned = cleaner.clean(section.text)
        if not cleaned:
            continue
        chunks = chunk_text(cleaned, cfg)
        all_chunks.extend(enrich(chunks, filing, section))

    serialized_tables = serializer.serialize_all(filing.raw_tables)
    for i, table_text in enumerate(serialized_tables):
        table_section = ParsedSection(
            item_label=f"Table {i}", title="Financial Table", text=table_text
        )
        chunks = chunk_text(table_text, cfg)
        all_chunks.extend(enrich(chunks, filing, table_section))

    return all_chunks


def main() -> None:
    """Load config, chunk every parsed filing on disk, write chunks.json, print summary."""
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    tickers = args.tickers or list(cfg.ingestion.tickers)
    years = args.years or list(cfg.ingestion.years)
    forms = args.forms or list(cfg.ingestion.filing_types)

    processed_dir = Path(cfg.ingestion.processed_data_dir)
    pairs = _find_parsed_filings(processed_dir, tickers, forms, years)
    logger.info(f"Found {len(pairs)} parsed filings on disk matching {forms} x {tickers} x {years}")

    cleaner = TextCleaner()
    serializer = TableSerializer()

    table = Table(title="Chunked Filings")
    table.add_column("Ticker")
    table.add_column("Form")
    table.add_column("Fiscal Year")
    table.add_column("Strategy")
    table.add_column("Chunks")
    table.add_column("Status")

    for parsed_path, out_path in pairs:
        filing = ParsedFiling.model_validate_json(parsed_path.read_text())
        if out_path.exists():
            table.add_row(
                filing.ticker, filing.form, str(filing.fiscal_year), "-", "-", "skipped (cached)"
            )
            continue

        chunks = chunk_filing(filing, cfg, cleaner, serializer)
        out_path.write_text(json.dumps([c.model_dump(mode="json") for c in chunks], indent=2))
        table.add_row(
            filing.ticker,
            filing.form,
            str(filing.fiscal_year),
            cfg.chunking.strategy,
            str(len(chunks)),
            "chunked",
        )

    Console().print(table)


if __name__ == "__main__":
    main()
