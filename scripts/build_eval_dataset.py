"""CLI: build data/eval/financebench.json from the public FinanceBench QA dataset.

Usage:
    python scripts/build_eval_dataset.py

WHY a build step instead of loading FinanceBench at eval time: FinanceBench
(PatronusAI/financebench on HuggingFace, free/public, 150 QA pairs) covers
84 companies' filings — almost none of which are actually ingested into this
project's local corpus. This script filters FinanceBench down to only the
questions whose source filing exists in data/processed/ (real chunks.json on
disk), so scripts/run_eval.py never tries to answer a question about a
document the pipeline was never given. Re-run this after ingesting any new
filing to pick up newly-eligible questions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.evaluation.models import EvalExample
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

# LEARN: FinanceBench's `company` field is a free-text display name, not a
# ticker — only companies actually in this project's universe need a mapping.
_COMPANY_TO_TICKER = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Alphabet": "GOOGL",
    "Google": "GOOGL",
    "Amazon": "AMZN",
}

_DOC_TYPE_TO_FORM = {"10k": "10-K", "10q": "10-Q"}


def _eligible_filings(processed_data_dir: str) -> set[tuple[str, str, int]]:
    """Scan data/processed/{ticker}/{form}_{year}/chunks.json for (ticker, form, year) on disk."""
    eligible = set()
    for chunks_path in Path(processed_data_dir).glob("*/*/chunks.json"):
        ticker = chunks_path.parent.parent.name
        form, _, year = chunks_path.parent.name.partition("_")
        if year.isdigit():
            eligible.add((ticker, form, int(year)))
    return eligible


def main() -> None:
    """Load config, filter FinanceBench to ingested filings, write data/eval/financebench.json."""
    cfg = load_config()
    eligible = _eligible_filings(cfg.ingestion.processed_data_dir)
    logger.info(f"Eligible local filings: {sorted(eligible)}")

    ds = load_dataset("PatronusAI/financebench", split="train")

    examples: list[EvalExample] = []
    for row in ds:
        ticker = _COMPANY_TO_TICKER.get(row["company"])
        form = _DOC_TYPE_TO_FORM.get(row["doc_type"])
        if ticker is None or form is None:
            continue
        if (ticker, form, row["doc_period"]) not in eligible:
            continue
        examples.append(
            EvalExample(
                financebench_id=row["financebench_id"],
                ticker=ticker,
                fiscal_year=row["doc_period"],
                doc_name=row["doc_name"],
                question=row["question"],
                ground_truth_answer=row["answer"],
                evidence_texts=[e["evidence_text"] for e in row["evidence"]],
            )
        )

    out_path = Path(cfg.evaluation.dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([e.model_dump() for e in examples], indent=2))

    table = Table(title="FinanceBench Examples Selected")
    table.add_column("Ticker")
    table.add_column("Doc")
    table.add_column("Count")
    for ticker, doc_name in sorted({(e.ticker, e.doc_name) for e in examples}):
        table.add_row(ticker, doc_name, str(sum(1 for e in examples if e.doc_name == doc_name)))
    console.print(table)
    logger.info(f"Wrote {len(examples)} eval examples to {out_path}")


if __name__ == "__main__":
    main()
