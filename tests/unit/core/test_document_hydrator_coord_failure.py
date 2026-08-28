"""
TDD tests proving the document hydrator survives a coord-cache read failure.

``hydrate_documents`` calls ``coord_cache.get_coords_batch`` exactly once per
file. Before the fix a read failure was silently swallowed; now
``get_coords_batch`` raises ``CoordCacheReadError`` and the hydrator MUST:
  * record ``doc.metadata["coord_cache_error"] = True`` for every chunk of the
    failed file, and
  * keep processing the OTHER files (it must not abort or raise uncaught).

T3 monkeypatches only ``coord_cache.get_coords_batch`` (the single caller site),
so no real DB / PDF I/O is required — keeping the test fast and deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from common.exceptions import CoordCacheReadError
from core.document_hydrator import hydrate_documents


class _FakeCoordCache:
    """Stand-in for the singleton ``coord_cache`` with an injectable failure."""

    def __init__(self, fail_hash: str) -> None:
        self._fail_hash = fail_hash

    async def get_coords_batch(
        self, file_hash: str, page_nums: list[int]
    ) -> dict[int, list[Any]]:
        if file_hash == self._fail_hash:
            raise CoordCacheReadError(
                f"injected cache read failure for {file_hash}",
                details={"file_hash": file_hash},
            )
        # Healthy file: return a plausible coordinate payload for each page.
        return {
            page: [{"x0": 1, "y0": 2, "x1": 3, "y1": 4, "w": "word"}]
            for page in page_nums
        }


def _make_doc(
    file_path: str, file_hash: str, page: int, has_coordinates: bool = True
) -> Document:
    return Document(
        page_content=f"content for {file_path} p{page}",
        metadata={
            "file_path": file_path,
            "file_hash": file_hash,
            "page": page,
            "pages": [page],
            "has_coordinates": has_coordinates,
        },
    )


@pytest.fixture
def patch_coord_cache(monkeypatch: Any) -> None:
    """Redirect ``document_hydrator.coord_cache`` to a controlled fake."""
    return monkeypatch


def test_t3_failed_file_marked_and_other_files_processed(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """T3: a read failure on one file is recorded, others are still hydrated."""
    good_path = str(tmp_path / "good.pdf")
    bad_path = str(tmp_path / "bad.pdf")

    # Create 2 files so ``os.path.exists`` checks in the hydrator pass.
    open(good_path, "w").close()
    open(bad_path, "w").close()

    fake = _FakeCoordCache(fail_hash="bad_hash")
    monkeypatch.setattr(
        "core.document_hydrator.coord_cache",
        fake,
        raising=True,
    )

    # Two chunks for the failing file, two chunks for the healthy file.
    bad_chunks = [
        _make_doc(bad_path, "bad_hash", 1),
        _make_doc(bad_path, "bad_hash", 2),
    ]
    good_chunks = [
        _make_doc(good_path, "good_hash", 1),
        _make_doc(good_path, "good_hash", 2),
    ]
    docs = [*bad_chunks, *good_chunks]

    # Must NOT raise; must process all files.
    import asyncio

    asyncio.run(hydrate_documents(docs))

    # (a) Failed-file chunks carry the explicit error marker.
    for doc in bad_chunks:
        assert doc.metadata.get("coord_cache_error") is True

    # (b) Healthy-file chunks were hydrated (coordinates attached).
    for doc in good_chunks:
        page = doc.metadata["page"]
        assert doc.metadata.get("coord_cache_error") is not True
        assert doc.metadata.get("page_coords") == {
            page: [{"x0": 1, "y0": 2, "x1": 3, "y1": 4, "w": "word"}]
        }
        assert doc.metadata.get("word_coords") is not None


def test_t4_coord_cache_error_survives_api_whitelist() -> None:
    """API 메타데이터 화이트리스트가 coord_cache_error 마커를 통과시키는지 확인.

    _doc_to_source는 클라이언트로 직렬화하기 전 메타데이터 키를 필터링하므로,
    이 마커가 누락되면 UI가 실패를 감지하지 못하고 하이라이트가 조용히 사라진다.
    """
    from api.api_server import _doc_to_source

    doc = Document(
        page_content="본문",
        metadata={"page": 1, "coord_cache_error": True, "file_hash": "h1"},
    )
    source = _doc_to_source(doc)
    assert source.get("coord_cache_error") is True
    assert source.get("file_hash") == "h1"
