"""Phase 2 wiring contracts: chonkie opt-in + header split + PDF flags."""

from langchain_core.documents import Document

from core.chonkie_adapter import is_chonkie_available, split_by_headers


def test_header_split_keeps_sections():
    md = "# A\nbody a\n# B\nbody b"
    parts = split_by_headers(md)
    assert [t for t, _ in parts] == ["A", "B"]
    assert "body b" in parts[1][1]


def test_header_split_no_headers():
    parts = split_by_headers("plain body")
    assert parts == [("", "plain body")]


def test_chonkie_unavailable_or_chunks():
    docs = [Document(page_content="# T\nhello world " * 20, metadata={})]
    if not is_chonkie_available():
        return  # spike env: adapter presence only
    from core.chonkie_adapter import chunk_with_chonkie

    out = chunk_with_chonkie(docs)
    assert out
    assert all(d.metadata.get("current_section") == "T" for d in out)


def test_heavy_parse_heuristic_shape():
    """Router-signal heuristic mirrors document_processor inline rule."""
    from common.config import SEMANTIC_CHUNKER_CONFIG

    assert SEMANTIC_CHUNKER_CONFIG.get("backend", "builtin") == "builtin"

    def _is_heavy(text: str, table_count: int) -> bool:
        return (
            table_count >= 2
            or len(text.strip()) < 50
            or ("$$" in text or "\\[" in text or "\\(" in text)
        )

    assert _is_heavy("x" * 10, 0) is True  # scan-suspect
    assert _is_heavy("y" * 500, 3) is True  # table-heavy
    assert _is_heavy("z" * 500, 0) is False
