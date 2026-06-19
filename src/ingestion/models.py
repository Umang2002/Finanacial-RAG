"""Pydantic models for SEC filing references and downloaded artifacts.

WHAT: FilingRef describes a filing as listed by SEC EDGAR's submissions API
(before download). DownloadedFiling extends that with where the document
landed on disk.
WHY: these cross the boundary between sec_loader.py, scripts/download_filings.py,
and later phases that read data/raw/*/metadata.json — pydantic gives
validation + JSON (de)serialization for free, per CLAUDE.md "use pydantic for
data structures that cross module boundaries".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

FilingForm = Literal["10-K", "10-Q"]


class FilingRef(BaseModel):
    """One filing entry from SEC EDGAR's submissions API, before download."""

    ticker: str
    cik: str = Field(..., description="10-digit zero-padded CIK, e.g. '0000320193'")
    form: FilingForm
    filing_date: date
    report_date: date
    accession_no: str = Field(..., description="e.g. '0000320193-23-000106'")
    primary_doc: str = Field(..., description="primary document filename, e.g. 'aapl-20230930.htm'")

    @property
    def fiscal_year(self) -> int:
        """Fiscal year derived from report_date (not filing_date).

        WHY: a 10-K covering FY2023 is often filed in early 2024 — grouping
        by filing_date would misfile it under 2024.
        """
        return self.report_date.year

    @property
    def accession_no_compact(self) -> str:
        """Accession number without dashes, as used in EDGAR archive URLs."""
        return self.accession_no.replace("-", "")


class DownloadedFiling(BaseModel):
    """A FilingRef plus where its document landed on disk — written as metadata.json."""

    ticker: str
    cik: str
    form: FilingForm
    filing_date: date
    report_date: date
    fiscal_year: int
    accession_no: str
    primary_doc: str
    download_url: str
    raw_path: Path
    metadata_path: Path
    size_bytes: int
    skipped: bool = Field(
        default=False,
        description="True if the file already existed and was not re-downloaded",
    )

    @classmethod
    def from_ref(
        cls,
        ref: FilingRef,
        download_url: str,
        raw_path: Path,
        metadata_path: Path,
        size_bytes: int,
        skipped: bool = False,
    ) -> DownloadedFiling:
        """Build a DownloadedFiling from a FilingRef plus the download result."""
        return cls(
            ticker=ref.ticker,
            cik=ref.cik,
            form=ref.form,
            filing_date=ref.filing_date,
            report_date=ref.report_date,
            fiscal_year=ref.fiscal_year,
            accession_no=ref.accession_no,
            primary_doc=ref.primary_doc,
            download_url=download_url,
            raw_path=raw_path,
            metadata_path=metadata_path,
            size_bytes=size_bytes,
            skipped=skipped,
        )


class ParsedSection(BaseModel):
    """One Item-numbered section of a 10-K/10-Q (e.g. 'Item 7. MD&A')."""

    item_label: str = Field(..., description="e.g. 'Item 1A'")
    title: str = Field(..., description="e.g. 'Risk Factors'")
    text: str


class ParsedFiling(BaseModel):
    """Clean structured output of HTMLFilingParser — input to Phase 2 chunkers.

    WHY raw_tables stays as raw HTML strings, not flattened text: table
    structure (rows/cols) is needed by table_serializer.py in Phase 2 to
    produce sentences like "Revenue: $383B (2023)" — flattening here would
    throw that structure away.
    """

    ticker: str
    cik: str
    form: FilingForm
    fiscal_year: int
    company_name: str
    period_end_date: date
    full_text: str
    sections: list[ParsedSection]
    raw_tables: list[str] = Field(default_factory=list)
