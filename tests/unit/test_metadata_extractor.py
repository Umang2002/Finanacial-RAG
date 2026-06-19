"""Unit tests for extract_metadata (src/ingestion/metadata_extractor.py)."""

from __future__ import annotations

from datetime import date

from bs4 import BeautifulSoup

from src.ingestion.metadata_extractor import extract_metadata

# Minimal inline-XBRL dei: tags, matching the shape confirmed against a real
# AAPL 10-K (see notebooks/01_data_exploration.ipynb section 7).
SAMPLE_HTML = """
<html><body>
<span name="dei:EntityRegistrantName">Apple Inc.</span>
<span name="dei:EntityCentralIndexKey">320193</span>
<span name="dei:DocumentFiscalYearFocus">2023</span>
<span name="dei:DocumentPeriodEndDate">September 30, 2023</span>
</body></html>
"""


def test_extract_metadata_reads_dei_tags() -> None:
    """All four dei: fields are read and typed correctly."""
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    meta = extract_metadata(soup, fallback_period_end=date(2023, 9, 30))

    assert meta["company_name"] == "Apple Inc."
    assert meta["cik"] == "0000320193"  # zero-padded to 10 digits
    assert meta["fiscal_year"] == 2023
    assert meta["period_end_date"] == date(2023, 9, 30)


def test_extract_metadata_falls_back_when_tags_missing() -> None:
    """Missing dei: tags fall back to defaults instead of raising."""
    soup = BeautifulSoup("<html><body>no xbrl here</body></html>", "lxml")
    meta = extract_metadata(soup, fallback_period_end=date(2022, 12, 31))

    assert meta["company_name"] == "UNKNOWN"
    assert meta["cik"] == ""
    assert meta["fiscal_year"] == 2022  # from fallback_period_end.year
    assert meta["period_end_date"] == date(2022, 12, 31)


def test_extract_metadata_falls_back_on_empty_period_end_text() -> None:
    """Empty DocumentPeriodEndDate text (common — value lives in iXBRL attrs) uses fallback."""
    html = """
    <span name="dei:EntityRegistrantName">Apple Inc.</span>
    <span name="dei:DocumentPeriodEndDate" format="ixt:date-monthname-day-year-en"></span>
    """
    soup = BeautifulSoup(html, "lxml")
    meta = extract_metadata(soup, fallback_period_end=date(2023, 9, 30))

    assert meta["period_end_date"] == date(2023, 9, 30)
