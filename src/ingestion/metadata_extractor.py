"""Extracts company name, CIK, fiscal year, and period-end date from SEC filings.

WHAT: reads inline-XBRL `dei:` (Document and Entity Information) tags —
EDGAR 10-K/10-Q filings since ~2019 embed these as <span name="dei:...">
or similar tags regardless of the surrounding HTML structure.
WHY separate from html_parser.py: dei: tags are structurally simple
key->text lookups, independent of the Item-section-splitting logic — keeping
this standalone makes it directly unit-testable and reusable if a PDF parser
is added later (XBRL-free filings would need a different extraction path,
but the output shape stays the same).
"""

from __future__ import annotations

from datetime import date, datetime

from bs4 import BeautifulSoup

# LEARN: confirmed against a real AAPL 10-K (see notebooks/01_data_exploration.ipynb,
# section 7) — these dei: tags are present and unambiguous. DocumentPeriodEndDate's
# rendered text can be empty (value lives in iXBRL @contextRef machinery instead),
# so we fall back to FilingRef.report_date for that one field.
_DEI_TAGS = {
    "company_name": "dei:EntityRegistrantName",
    "cik": "dei:EntityCentralIndexKey",
    "fiscal_year": "dei:DocumentFiscalYearFocus",
    "period_end_date": "dei:DocumentPeriodEndDate",
}


def extract_metadata(soup: BeautifulSoup, fallback_period_end: date) -> dict:
    """Pull dei: XBRL fields out of a parsed filing into a flat dict.

    Args:
        soup: BeautifulSoup tree of the filing's raw HTML.
        fallback_period_end: used for period_end_date when the dei: tag's
            text is empty (common — EDGAR sometimes encodes the value as an
            iXBRL attribute, not visible text).

    Returns:
        dict with keys company_name (str), cik (str), fiscal_year (int),
        period_end_date (date).
    """
    values: dict[str, str] = {}
    for field, dei_name in _DEI_TAGS.items():
        tag = soup.find(attrs={"name": dei_name})
        values[field] = tag.get_text(strip=True) if tag else ""

    period_end_date = _parse_period_end(values["period_end_date"]) or fallback_period_end

    return {
        "company_name": values["company_name"] or "UNKNOWN",
        "cik": values["cik"].zfill(10) if values["cik"] else "",
        "fiscal_year": int(values["fiscal_year"])
        if values["fiscal_year"].isdigit()
        else fallback_period_end.year,
        "period_end_date": period_end_date,
    }


def _parse_period_end(text: str) -> date | None:
    """Parse dei:DocumentPeriodEndDate's rendered text, e.g. 'September 30, 2023'.

    Returns None if text is empty/unparseable — caller falls back to report_date.
    """
    if not text:
        return None
    try:
        return datetime.strptime(text, "%B %d, %Y").date()
    except ValueError:
        return None
