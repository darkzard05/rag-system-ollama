"""병렬 하이드레이션 (B12) 검증 테스트.

``hydrate_documents``는 파일의 페이지들을 ``asyncio.gather``로 병렬
캐시-읽기/추출하며, ``coord_cache.save_coords``는 신선하게 추출된
(fresh, 캐시 미스) 페이지에만 호출되어야 한다. 캐시 적중 페이지를
재저장하면 이전 추출의 더 풍부한 데이터를 덮어쓰고 불필요한 I/O가 된다.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from core import document_hydrator
from core.document_hydrator import hydrate_documents

_EXTRACTED = [("a", 1.0, 2.0, 3.0, 4.0), ("b", 5.0, 6.0, 7.0, 8.0)]
_CACHED = [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "w": "cached"}]


def _make_doc(path: str, file_hash: str, pages: list[int]) -> Document:
    return Document(
        page_content=f"content for {path}",
        metadata={
            "file_path": path,
            "file_hash": file_hash,
            "page": pages[0],
            "pages": pages,
            "has_coordinates": True,
        },
    )


@pytest.mark.asyncio
async def test_parallel_hydration_saves_coords_only_for_fresh_pages(
    tmp_path: Path,
) -> None:
    """캐시 적중 페이지는 재저장하지 않고, 신선 추출 페이지만 저장한다."""
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), "h1", [1, 2, 3])

    cache_hits = {2: _CACHED}
    save_coords = AsyncMock(return_value=True)

    with (
        patch.object(
            document_hydrator.coord_cache,
            "get_coords_batch",
            return_value=cache_hits,
        ),
        patch.object(
            document_hydrator,
            "_extract_page_words_sync",
            return_value=_EXTRACTED,
        ),
        patch.object(
            document_hydrator.coord_cache,
            "save_coords",
            save_coords,
        ),
    ):
        await hydrate_documents([doc])

    saved_pages = {call.args[1] for call in save_coords.call_args_list}
    assert saved_pages == {1, 3}
    for call in save_coords.call_args_list:
        assert call.args[0] == "h1"
        assert call.args[2] == _EXTRACTED

    assert set(doc.metadata["page_coords"]) == {1, 2, 3}
    assert doc.metadata["page_coords"][2] == _CACHED


@pytest.mark.asyncio
async def test_parallel_hydration_extraction_failure_does_not_raise(
    tmp_path: Path,
) -> None:
    """추출 실패 페이지는 예외 없이 좌표 없이 남고, save_coords도 호출 안 됨."""
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), "h2", [1, 2])

    cache_hits = {2: _CACHED}
    save_coords = AsyncMock(return_value=True)

    with (
        patch.object(
            document_hydrator.coord_cache,
            "get_coords_batch",
            return_value=cache_hits,
        ),
        patch.object(
            document_hydrator,
            "_extract_page_words_sync",
            return_value=None,
        ),
        patch.object(
            document_hydrator.coord_cache,
            "save_coords",
            save_coords,
        ),
    ):
        await hydrate_documents([doc])

    save_coords.assert_not_called()
    assert doc.metadata["page_coords"] == {2: _CACHED}


@pytest.mark.asyncio
async def test_parallel_hydration_keeps_pages_order(tmp_path: Path) -> None:
    """gather 완료 순서와 무관하게 page_coords 키는 원본 pages 순서를 유지."""
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), "h3", [1, 2, 3])

    cache_hits = {2: _CACHED}

    def _slow_first(path: str, page_num: int, chunk_bbox: object) -> list:
        if page_num == 1:
            time.sleep(0.05)
        return _EXTRACTED

    with (
        patch.object(
            document_hydrator.coord_cache,
            "get_coords_batch",
            return_value=cache_hits,
        ),
        patch.object(
            document_hydrator,
            "_extract_page_words_sync",
            side_effect=_slow_first,
        ),
        patch.object(
            document_hydrator.coord_cache,
            "save_coords",
            AsyncMock(return_value=True),
        ),
    ):
        await hydrate_documents([doc])

    assert list(doc.metadata["page_coords"].keys()) == [1, 2, 3]
