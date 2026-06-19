"""Parses PDFs with PyMuPDF + pdfplumber; extracts text blocks and tables.

WHY deferred: SEC EDGAR's primary 10-K/10-Q documents have been inline-XBRL
.htm files since ~2002 (confirmed for AAPL filings in
notebooks/01_data_exploration.ipynb) — there is no PDF in our actual
ingestion path (see html_parser.py). Not implementing this until a real
need shows up (e.g. exhibits only available as PDF) avoids building unused
code, per CLAUDE.md's no-speculative-abstraction rule.
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(path: Path) -> None:
    """Placeholder — raises until a real PDF ingestion need exists."""
    raise NotImplementedError(
        "PDF parsing not needed yet — EDGAR 10-K/10-Q primary docs are inline-XBRL HTML."
    )
