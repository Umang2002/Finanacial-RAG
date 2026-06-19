"""Unit tests for TableSerializer (src/processing/table_serializer.py)."""

from __future__ import annotations

from src.processing.table_serializer import TableSerializer

SAMPLE_TABLE = """
<table>
<tr><td></td><td>2023</td><td>2022</td></tr>
<tr><td>Net sales</td><td>$383,285</td><td>$394,328</td></tr>
<tr><td>Operating income</td><td>$114,301</td><td>$119,437</td></tr>
</table>
"""


def test_serialize_produces_markdown_table() -> None:
    out = TableSerializer().serialize(SAMPLE_TABLE)
    assert "| Net sales | $383,285 | $394,328 |" in out
    assert "| --- | --- | --- |" in out


def test_serialize_produces_natural_language_sentences() -> None:
    out = TableSerializer().serialize(SAMPLE_TABLE)
    assert "Net sales (2023): $383,285" in out
    assert "Operating income (2022): $119,437" in out


def test_serialize_empty_table_returns_empty_string() -> None:
    assert TableSerializer().serialize("<table></table>") == ""
    assert TableSerializer().serialize("<table><tr><td>only header</td></tr></table>") == ""


def test_serialize_all_drops_unparseable_tables() -> None:
    out = TableSerializer().serialize_all([SAMPLE_TABLE, "<table></table>"])
    assert len(out) == 1
