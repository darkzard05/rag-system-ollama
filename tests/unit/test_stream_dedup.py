"""Unit tests for streaming document dedup helper in core.rag_core."""

from __future__ import annotations

from core.rag_core import _dedup_docs


class _FakeDoc:
    """Minimal stand-in for a langchain Document with metadata/page_content."""

    def __init__(self, page_content: str, doc_id: str | None = None) -> None:
        self.page_content = page_content
        self.metadata = {"doc_id": doc_id} if doc_id is not None else {}


def test_dedup_unique_per_rewrite() -> None:
    """Same doc across two retrieve chunks yields a single kept doc."""
    seen: set[str] = set()
    doc = _FakeDoc("identical content", doc_id="abc")

    first = _dedup_docs([doc], seen)
    assert len(first) == 1

    # Second retrieve chunk re-runs with the same document.
    second = _dedup_docs([doc], seen)
    assert second == []


def test_dedup_distinct_docs_kept() -> None:
    """Distinct docs are all kept (fallback content hash path)."""
    seen: set[str] = set()
    a = _FakeDoc("content A")
    b = _FakeDoc("content B")

    kept = _dedup_docs([a, b], seen)
    assert kept == [a, b]
    assert len(seen) == 2


def test_empty_docs_no_error() -> None:
    """Empty input returns empty list without raising."""
    assert _dedup_docs([], set()) == []


def test_dedup_missing_metadata_safe() -> None:
    """Docs lacking metadata/page_content don't crash."""
    seen: set[str] = set()

    class _Bare:
        pass

    kept = _dedup_docs([_Bare(), _Bare()], seen)
    # Both have empty content -> same hash -> only one kept.
    assert len(kept) == 1
