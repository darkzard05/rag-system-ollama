"""Truncated JSON recovery must keep arbitrary document ids intact."""

from __future__ import annotations

from core.graph._graph_core import _recover_citations


def test_recover_citation_keeps_hyphens_and_underscores() -> None:
    """A stable id with punctuation survives a broken JSON tail."""
    blob = (
        '{"final_answer": "답변", "citations": ['
        '{"doc_id": "abc-123_chunk_4", "anchor": "핵심 근거"}'
    )

    assert _recover_citations(blob) == [
        {"doc_id": "abc-123_chunk_4", "anchor": "핵심 근거"}
    ]


def test_recover_citation_still_handles_hex_and_numeric_ids() -> None:
    """Existing short ids keep their current recovery behavior."""
    hex_blob = '{"citations": [{"doc_id": "8f3a1c2d7b4e", "anchor": "수치"}]'
    numeric_blob = '{"citations": [{"doc_id": 3, "anchor": "수치"}]'

    assert _recover_citations(hex_blob) == [
        {"doc_id": "8f3a1c2d7b4e", "anchor": "수치"}
    ]
    assert _recover_citations(numeric_blob) == [{"doc_id": "3", "anchor": "수치"}]
