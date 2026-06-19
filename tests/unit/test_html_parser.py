"""Unit tests for HTMLFilingParser (src/ingestion/html_parser.py).

WHY a synthetic fixture instead of a real downloaded filing: keeps tests
offline/fast and lets us exercise the ToC-vs-real-section heuristic with a
known, small ground truth. The shape (tightly-packed ToC entries, then a
long preamble, then real Item bodies) mirrors what notebooks/01_data_exploration.ipynb
found in a real AAPL 10-K.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.ingestion.html_parser import HTMLFilingParser
from src.ingestion.models import FilingRef

PREAMBLE = (
    "This Annual Report on Form 10-K contains forward-looking statements within the "
    "meaning of applicable securities laws regarding future events and the financial "
    "performance of the Company that involve risks and uncertainties. "
)

SAMPLE_HTML = f"""
<html>
<head>
<span name="dei:EntityRegistrantName">Widget Corp</span>
<span name="dei:EntityCentralIndexKey">320193</span>
<span name="dei:DocumentFiscalYearFocus">2023</span>
<span name="dei:DocumentPeriodEndDate">September 30, 2023</span>
</head>
<body>
<span>Item 1. Business 1 Item 1A. Risk Factors 5 Item 2. Properties 17</span>
<span>{PREAMBLE}</span>
<span>Item 1. Business The Company designs and sells widgets globally, including
hardware, software, and related services to consumers worldwide.</span>
<span>Item 1A. Risk Factors Our business faces many risks including intense
competition, component shortages, and currency fluctuations.</span>
<table><tr><td>Revenue</td><td>383,285</td></tr></table>
</body>
</html>
"""


@pytest.fixture
def ref() -> FilingRef:
    return FilingRef(
        ticker="WDGT",
        cik="0000320193",
        form="10-K",
        filing_date=date(2023, 11, 3),
        report_date=date(2023, 9, 30),
        accession_no="0000320193-23-000106",
        primary_doc="wdgt-20230930.htm",
    )


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    path = tmp_path / "primary.htm"
    path.write_text(SAMPLE_HTML)
    return path


def test_parse_drops_toc_and_keeps_real_sections(raw_path: Path, ref: FilingRef) -> None:
    """The tightly-packed ToC entries are dropped; only the two real sections remain."""
    parsed = HTMLFilingParser().parse(raw_path, ref)

    assert [s.item_label for s in parsed.sections] == ["Item 1", "Item 1A"]
    assert "widgets" in parsed.sections[0].text
    assert "Item 1A" not in parsed.sections[0].text  # section boundary cuts cleanly
    assert "competition" in parsed.sections[1].text


def test_parse_extracts_metadata(raw_path: Path, ref: FilingRef) -> None:
    """dei: XBRL tags populate company_name/cik/fiscal_year/period_end_date."""
    parsed = HTMLFilingParser().parse(raw_path, ref)

    assert parsed.company_name == "Widget Corp"
    assert parsed.cik == "0000320193"
    assert parsed.fiscal_year == 2023
    assert parsed.period_end_date == date(2023, 9, 30)


def test_parse_extracts_raw_tables(raw_path: Path, ref: FilingRef) -> None:
    """Tables are kept as raw HTML strings, not flattened into full_text loss."""
    parsed = HTMLFilingParser().parse(raw_path, ref)

    assert len(parsed.raw_tables) == 1
    assert "383,285" in parsed.raw_tables[0]
    assert parsed.raw_tables[0].strip().startswith("<table>")


def test_parse_with_no_item_headers_returns_single_section(tmp_path: Path, ref: FilingRef) -> None:
    """A document with no 'Item N.' pattern falls back to one whole-document section."""
    path = tmp_path / "primary.htm"
    path.write_text("<html><body><span>No items here, just prose.</span></body></html>")

    parsed = HTMLFilingParser().parse(path, ref)

    assert len(parsed.sections) == 1
    assert parsed.sections[0].item_label == "N/A"
