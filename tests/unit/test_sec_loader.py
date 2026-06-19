"""Unit tests for SECEdgarLoader (src/ingestion/sec_loader.py).

WHY: all SEC EDGAR HTTP calls are mocked via httpx.MockTransport — these
tests run offline, fast, and don't burn SEC's rate limit. Live integration
is exercised separately via `python scripts/download_filings.py`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from src.ingestion.models import FilingRef
from src.ingestion.sec_loader import (
    _SUBMISSIONS_URL,
    _TICKERS_URL,
    SECEdgarLoader,
)

TEST_USER_AGENT = "Test Suite test@example.com"

# Minimal company_tickers.json shape: dict-of-dicts keyed by index strings.
TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

# Minimal submissions JSON: filings.recent holds parallel arrays.
SUBMISSIONS_JSON = {
    "cik": "320193",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-23-000106",  # 10-K, report date 2023-09-30
                "0000320193-23-000077",  # 10-Q, report date 2023-07-01
                "0000320193-22-000108",  # 10-K, report date 2022-09-24
            ],
            "filingDate": ["2023-11-03", "2023-08-04", "2022-10-28"],
            "reportDate": ["2023-09-30", "2023-07-01", "2022-09-24"],
            "form": ["10-K", "10-Q", "10-K"],
            "primaryDocument": [
                "aapl-20230930.htm",
                "aapl-20230701.htm",
                "aapl-20220924.htm",
            ],
        }
    },
}

DOC_CONTENT = b"<html><body>Apple 10-K FY2023</body></html>"


def make_loader(tmp_path: Path, call_log: list[str]) -> SECEdgarLoader:
    """Build a SECEdgarLoader wired to a MockTransport that records request URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        call_log.append(url)
        if url == _TICKERS_URL:
            return httpx.Response(200, json=TICKERS_JSON)
        if url == _SUBMISSIONS_URL.format(cik="0000320193"):
            return httpx.Response(200, json=SUBMISSIONS_JSON)
        if url.startswith("https://www.sec.gov/Archives/edgar/data/"):
            return httpx.Response(200, content=DOC_CONTENT)
        return httpx.Response(404)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": TEST_USER_AGENT}
    )
    return SECEdgarLoader(user_agent=TEST_USER_AGENT, raw_data_dir=tmp_path, client=client)


def make_ref(accession: str, primary_doc: str, report_date: str, filing_date: str) -> FilingRef:
    """Build a FilingRef for AAPL's 10-K FY2023, matching SUBMISSIONS_JSON entry 0."""
    return FilingRef(
        ticker="AAPL",
        cik="0000320193",
        form="10-K",
        filing_date=date.fromisoformat(filing_date),
        report_date=date.fromisoformat(report_date),
        accession_no=accession,
        primary_doc=primary_doc,
    )


def test_invalid_user_agent_raises(tmp_path: Path) -> None:
    """User-Agent without an '@' (no email) should fail fast, not at request time."""
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        SECEdgarLoader(user_agent="NoEmailHere", raw_data_dir=tmp_path)


def test_resolve_cik_caches(tmp_path: Path) -> None:
    """resolve_cik maps ticker -> 10-digit CIK and fetches company_tickers.json once."""
    call_log: list[str] = []
    loader = make_loader(tmp_path, call_log)

    assert loader.resolve_cik("aapl") == "0000320193"
    assert loader.resolve_cik("AAPL") == "0000320193"

    assert call_log.count(_TICKERS_URL) == 1  # cached after first call


def test_resolve_cik_unknown_ticker(tmp_path: Path) -> None:
    """Unknown ticker raises a clear ValueError instead of returning None."""
    loader = make_loader(tmp_path, [])
    with pytest.raises(ValueError, match="ZZZZ"):
        loader.resolve_cik("ZZZZ")


def test_list_filings_filters_form_and_year(tmp_path: Path) -> None:
    """Only 10-K filings whose report_date falls in 2023 are returned."""
    loader = make_loader(tmp_path, [])
    refs = loader.list_filings("AAPL", forms=["10-K"], years=[2023])

    assert len(refs) == 1
    assert refs[0].accession_no == "0000320193-23-000106"
    assert refs[0].fiscal_year == 2023


def test_list_filings_multiple_forms_and_years(tmp_path: Path) -> None:
    """10-K + 10-Q across two years returns all three sample filings."""
    loader = make_loader(tmp_path, [])
    refs = loader.list_filings("AAPL", forms=["10-K", "10-Q"], years=[2022, 2023])

    assert len(refs) == 3
    assert {r.fiscal_year for r in refs} == {2022, 2023}


def test_download_filing_writes_document_and_metadata(tmp_path: Path) -> None:
    """download_filing saves the primary doc + a metadata.json with matching fields."""
    loader = make_loader(tmp_path, [])
    ref = make_ref("0000320193-23-000106", "aapl-20230930.htm", "2023-09-30", "2023-11-03")

    result = loader.download_filing(ref)

    assert result.raw_path.exists()
    assert result.raw_path.read_bytes() == DOC_CONTENT
    assert result.skipped is False
    assert result.fiscal_year == 2023

    meta = json.loads(result.metadata_path.read_text())
    assert meta["cik"] == "0000320193"
    assert meta["accession_no"] == "0000320193-23-000106"
    assert meta["form"] == "10-K"


def test_download_filing_is_idempotent(tmp_path: Path) -> None:
    """Re-downloading the same filing skips the network call and marks skipped=True."""
    call_log: list[str] = []
    loader = make_loader(tmp_path, call_log)
    ref = make_ref("0000320193-23-000106", "aapl-20230930.htm", "2023-09-30", "2023-11-03")

    first = loader.download_filing(ref)
    archive_calls_before = sum(1 for u in call_log if "/Archives/" in u)

    second = loader.download_filing(ref)
    archive_calls_after = sum(1 for u in call_log if "/Archives/" in u)

    assert first.skipped is False
    assert second.skipped is True
    assert archive_calls_after == archive_calls_before  # no re-download
    assert second.raw_path.read_bytes() == DOC_CONTENT


def test_download_all_returns_filings_for_each_ticker(tmp_path: Path) -> None:
    """download_all lists + downloads filings across all configured tickers."""
    loader = make_loader(tmp_path, [])

    results = loader.download_all(["AAPL"], forms=["10-K", "10-Q"], years=[2022, 2023])

    assert len(results) == 3
    assert all(r.raw_path.exists() for r in results)
