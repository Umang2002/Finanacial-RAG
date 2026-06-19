"""Normalizes whitespace and removes boilerplate headers/footers from extracted text.

WHAT: TextCleaner.clean() runs on ParsedFiling.full_text / ParsedSection.text
before chunking.
WHY narrow in scope: html_parser.py already collapses all whitespace to
single spaces (`re.sub(r"\\s+", " ", ...)`) and strips the table-of-contents
block via its gap heuristic — so this layer isn't re-doing that. What's left
after HTML parsing: stray Unicode artifacts (non-breaking spaces, smart
quotes) and the literal "Table of Contents" string that sometimes survives
as an inline cross-reference (e.g. "see Table of Contents above"). We
deliberately do NOT try to strip standalone page-number digits — those are
indistinguishable from real dollar figures once everything is flattened to
single-line text, and a false positive there silently corrupts financial
data.
"""

from __future__ import annotations

import re
import unicodedata

from src.utils.logging import get_logger

logger = get_logger(__name__)

_TOC_RE = re.compile(r"\bTable of Contents\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

# LEARN: smart quotes/dashes from Word-exported filings render as separate
# glyphs that don't match plain ASCII in downstream regex/NL serialization —
# normalize to ASCII equivalents rather than leaving mixed encodings.
_UNICODE_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "\xa0": " ",
}


class TextCleaner:
    """Strips recurring SEC boilerplate and normalizes Unicode before chunking."""

    def clean(self, text: str) -> str:
        """Normalize Unicode, drop stray 'Table of Contents' mentions, collapse whitespace.

        Returns "" for empty/whitespace-only input rather than raising —
        callers (chunkers) already treat "" as "nothing to chunk".
        """
        if not text or not text.strip():
            return ""

        for old, new in _UNICODE_REPLACEMENTS.items():
            text = text.replace(old, new)
        text = unicodedata.normalize("NFKC", text)
        text = _TOC_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text)
        return text.strip()
