"""Unit tests for TextCleaner (src/processing/cleaner.py)."""

from __future__ import annotations

from src.processing.cleaner import TextCleaner


def test_clean_empty_text_returns_empty_string() -> None:
    assert TextCleaner().clean("") == ""
    assert TextCleaner().clean("   \n  ") == ""


def test_clean_normalizes_unicode_quotes_and_dashes() -> None:
    text = "The Company’s revenue – net of returns — grew."
    cleaned = TextCleaner().clean(text)
    assert "’" not in cleaned
    assert "–" not in cleaned
    assert "—" not in cleaned
    assert "Company's revenue - net of returns - grew." in cleaned


def test_clean_strips_table_of_contents_mentions() -> None:
    text = "See Table of Contents for details. Revenue grew 10%."
    cleaned = TextCleaner().clean(text)
    assert "Table of Contents" not in cleaned
    assert "Revenue grew 10%." in cleaned


def test_clean_collapses_repeated_whitespace() -> None:
    text = "Revenue   grew\n\n10%   year over year."
    cleaned = TextCleaner().clean(text)
    assert cleaned == "Revenue grew 10% year over year."


def test_clean_preserves_dollar_figures() -> None:
    text = "Net sales were $383,285 million in fiscal 2023."
    cleaned = TextCleaner().clean(text)
    assert "$383,285" in cleaned
