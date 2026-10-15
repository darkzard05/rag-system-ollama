"""Regression locks for PoC-1/2/3 critical fixes."""

import tempfile

from langchain_core.documents import Document

from core.graph._generate import _docs_fingerprint, resolve_cached_context


def _doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"page": page})


def test_stale_context_rejected_without_fingerprint() -> None:
    docs = [_doc("new content", page=9)]
    assert resolve_cached_context({"context": "old context"}, docs) is None


def test_stale_context_rejected_on_mismatch() -> None:
    docs = [_doc("berlin", page=9)]
    state = {
        "context": "paris context",
        "context_fingerprint": _docs_fingerprint([_doc("paris", page=1)]),
    }
    assert resolve_cached_context(state, docs) is None


def test_cached_context_hit_on_same_docs() -> None:
    docs = [_doc("same", page=3)]
    state = {
        "context": "CTX",
        "context_fingerprint": _docs_fingerprint(docs),
    }
    assert resolve_cached_context(state, docs) == "CTX"


def test_file_hash_traversal_rejected() -> None:
    from cache.vector_cache import VectorStoreCache

    with tempfile.TemporaryDirectory() as tmp:
        for evil in ("../../evil", "/abs/path", "a/b", "..", "x.y"):
            try:
                VectorStoreCache("dummy.pdf", "model-x", cache_dir=tmp, file_hash=evil)
            except ValueError:
                continue
            raise AssertionError(f"traversal accepted: {evil!r}")
        cache = VectorStoreCache(
            "dummy.pdf",
            "model-x",
            cache_dir=tmp,
            file_hash="abc123_def-456",
        )
        import os

        assert os.path.commonpath(
            [os.path.abspath(tmp), os.path.abspath(cache.cache_dir)]
        ) == os.path.abspath(tmp)
