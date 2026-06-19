"""Converts extracted financial tables (raw HTML) to markdown + natural-language text for embedding.

WHAT: TableSerializer.serialize() turns one ParsedFiling.raw_tables entry
(a raw HTML <table> string) into a markdown table plus row-wise NL sentences
like "Net sales (2023): $383,285".
WHY both forms: markdown preserves row/column structure for an LLM reading
the chunk at generation time; the NL sentences exist because dense embedding
models match natural-language queries ("what was net sales in 2023?") far
better against prose than against pipe-delimited table syntax — BGE-M3 was
never trained to treat "| Net sales | $383,285 |" as semantically equivalent
to a sentence.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_rows(raw_html_table: str) -> list[list[str]]:
    """Parse a raw <table> HTML string into a list of text cells per row.

    WHY keep empty cells (don't filter them out): EDGAR tables commonly have
    a blank header cell above the row-label column (e.g. the corner cell
    above "Net sales" / left of "2023"), and blank data cells where a line
    item has no value for a period. Dropping empty cells would shift every
    later cell in that row left, misaligning the header-to-value zip in
    _to_natural_language(). Only fully-blank rows are dropped.
    """
    soup = BeautifulSoup(raw_html_table, "lxml")
    rows: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    return rows


def _to_markdown(rows: list[list[str]]) -> str:
    """Render extracted rows as a markdown table, using the first row as header."""
    header, *body = rows
    width = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        padded = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def _to_natural_language(rows: list[list[str]]) -> str:
    """Render each body row as "<label> (<column header>): <value>" sentences.

    WHY this shape works for EDGAR tables: financial statement tables
    consistently put the line-item label in column 0 and period values
    (e.g. fiscal years or quarters) in the header row — confirmed in
    notebooks/01_data_exploration.ipynb. Tables that don't fit this shape
    (e.g. no header row, single column) just produce no sentences here —
    the markdown form still carries their content.
    """
    header, *body = rows
    if len(header) < 2:
        return ""

    sentences: list[str] = []
    for row in body:
        if not row or not row[0]:
            continue
        label = row[0]
        for col_idx in range(1, min(len(header), len(row))):
            value = row[col_idx]
            if value:
                sentences.append(f"{label} ({header[col_idx]}): {value}")
    return ". ".join(sentences)


class TableSerializer:
    """Converts raw HTML <table> strings into markdown + natural-language text."""

    def serialize(self, raw_html_table: str) -> str:
        """Return markdown table + NL sentences for one raw HTML table, or "" if unparseable."""
        rows = _extract_rows(raw_html_table)
        if len(rows) < 2:
            return ""

        markdown = _to_markdown(rows)
        natural_language = _to_natural_language(rows)
        return f"{markdown}\n\n{natural_language}" if natural_language else markdown

    def serialize_all(self, raw_tables: list[str]) -> list[str]:
        """Serialize every table, dropping ones that produced no usable output."""
        serialized = [self.serialize(t) for t in raw_tables]
        return [s for s in serialized if s]
