"""Phase3 — chonkie adapter pure-unit tests (no Ollama needed)."""

import pytest
from langchain_core.documents import Document

chonkie = pytest.importorskip("chonkie")

from core.chonkie_adapter import split_by_headers  # noqa: E402


def _doc(text: str, section: str = "") -> Document:
    return Document(page_content=text, metadata={"current_section": section})


def test_split_by_headers_basic():
    md = "# INTRO\nbody one\n## SUB\nbody two"
    parts = split_by_headers(md)
    assert [t for t, _ in parts] == ["INTRO", "SUB"]
    assert "body one" in parts[0][1]


def test_split_by_headers_no_header():
    assert split_by_headers("plain text") == [("", "plain text")]


def test_postprocess_ref_via_section():
    from core.chunking import _postprocess_metadata

    docs = [
        _doc("some conclusion text here", "8 CONCLUSION"),
        _doc("- Author A, Title, arXiv", "REFERENCES"),
        _doc("- Author B, Title, NeurIPS", "REFERENCES"),
    ]
    _postprocess_metadata(docs)
    assert docs[0].metadata["is_reference"] is False
    assert docs[1].metadata["is_reference"] is True
    assert docs[2].metadata["is_reference"] is True
    assert docs[1].metadata["section_path"] == "REFERENCES"


def test_chonkie_partial_defaults():
    from core.chunking import _get_chonkie_chunk_fn

    fn = _get_chonkie_chunk_fn()
    # Phase4 sweep winner (thr 0.8/win 3/skip 0) — config 고정값과 일치해야 한다.
    assert fn.keywords["threshold"] == 0.8
    assert fn.keywords["similarity_window"] == 3
    assert fn.keywords["skip_window"] == 0
    assert fn.keywords["chunk_size"] == 512
