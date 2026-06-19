"""Parses SEC HTML filings (inline XBRL) with BeautifulSoup; extracts structured sections.

WHAT: HTMLFilingParser turns a raw downloaded 10-K/10-Q .htm file into a
ParsedFiling — clean full text, Item-numbered sections, and raw tables.
WHY: confirmed via notebooks/01_data_exploration.ipynb section 1 that EDGAR
filings are inline XBRL, not plain HTML — content lives in <span> tags with
XBRL attributes, not <p>/<h1-4> tags (both counts were 0 on a real AAPL
10-K). soup.get_text() handles this correctly since it walks all text nodes
regardless of tag type; the rest of this module is about splitting that flat
text into Item-numbered sections and keeping tables structured.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.models import FilingRef, ParsedFiling, ParsedSection
from src.utils.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# LEARN: `\s*` before the period — real AAPL ToC renders "Item 12 ." with a
# stray space (iXBRL splits the number and period into separate spans for
# some link-wrapped ToC rows). Without it, those two rows go unmatched,
# leaving an inflated ~230-char gap that the ToC-detection heuristic below
# misreads as the ToC/content boundary, cutting off too early.
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-C]?)\s*\.\s+")

# LEARN: a 10-K's table of contents lists every Item back-to-back with only
# a title + page number between them (~10-40 chars) — confirmed in notebook
# section 2 ("Business 1 Item 1A. Risk Factors 5 ..."). Real section bodies
# are separated by full paragraphs (hundreds-thousands of chars), except the
# transition from the last ToC entry into the preamble before Item 1's real
# heading, which is reliably >200 chars. Scanning for the first inter-match
# gap exceeding this threshold finds where the ToC ends and real content
# begins. Limitation: assumes one contiguous ToC near the top of the
# document — true for every EDGAR 10-K/10-Q observed, not guaranteed by spec.
_TOC_GAP_THRESHOLD = 200


def _split_sections(full_text: str) -> list[ParsedSection]:
    """Split full_text into Item-numbered sections, dropping the table of contents.

    WHY no dedup-by-label: 10-Q filings reuse Item numbers across Part I and
    Part II (e.g. Item 1 = "Financial Statements" in Part I, but Part II
    Item 1 = "Legal Proceedings"). Keeping every match in document order
    (instead of deduping by label) handles this for free — both show up as
    separate ParsedSection entries with the same item_label, which is fine
    since chunking/citation only needs label + text, not label uniqueness.
    """
    matches = [(m.group(1), m.start(), m.end()) for m in _ITEM_RE.finditer(full_text)]
    if not matches:
        return [ParsedSection(item_label="N/A", title="Full Document", text=full_text)]

    gaps = [matches[i + 1][1] - matches[i][1] for i in range(len(matches) - 1)]
    toc_end = next((i for i, gap in enumerate(gaps) if gap >= _TOC_GAP_THRESHOLD), None)
    real_matches = matches if toc_end is None else matches[toc_end + 1 :]

    sections = []
    for i, (label, start, content_start) in enumerate(real_matches):
        end = real_matches[i + 1][1] if i + 1 < len(real_matches) else len(full_text)
        # LEARN: title is best-effort (first 8 words after the label) — real
        # filings don't reliably put a sentence boundary right after the
        # title, so this can spill into body text. Fine here since the title
        # is descriptive metadata only; `text` (the field chunkers consume)
        # is sliced precisely between section start positions.
        title = " ".join(full_text[content_start:end].split()[:8])
        sections.append(
            ParsedSection(
                item_label=f"Item {label}", title=title, text=full_text[start:end].strip()
            )
        )
    return sections


class HTMLFilingParser:
    """Parses a downloaded SEC filing .htm file into a ParsedFiling."""

    def parse(self, raw_path: Path, ref: FilingRef) -> ParsedFiling:
        """Read raw_path, strip non-content tags, split into sections, return ParsedFiling.

        Args:
            raw_path: path to the downloaded primary.htm (from SECEdgarLoader).
            ref: FilingRef this document was downloaded for — supplies
                ticker/cik/form/fiscal_year/report_date as a fallback when
                XBRL metadata extraction can't find a value.
        """
        with open(raw_path, "rb") as f:
            soup = BeautifulSoup(f, "lxml")

        # LEARN: <head> holds XBRL schema/context metadata (URIs, not content);
        # script/style produce no readable text but get_text() would still
        # walk them — decomposing avoids noise in full_text.
        for tag in soup.find_all(["script", "style", "head"]):
            tag.decompose()

        raw_tables = [str(table) for table in soup.find_all("table")]
        full_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        sections = _split_sections(full_text)
        meta = extract_metadata(soup, fallback_period_end=ref.report_date)

        logger.info(
            f"Parsed {ref.ticker} {ref.form} {ref.fiscal_year}: "
            f"{len(full_text):,} chars, {len(sections)} sections, {len(raw_tables)} tables"
        )

        return ParsedFiling(
            ticker=ref.ticker,
            cik=ref.cik,
            form=ref.form,
            fiscal_year=meta["fiscal_year"],
            company_name=meta["company_name"],
            period_end_date=meta["period_end_date"],
            full_text=full_text,
            sections=sections,
            raw_tables=raw_tables,
        )
