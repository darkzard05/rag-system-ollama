"""Citation-coordinate fallback tests for document hydrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from core import document_hydrator
from core.document_hydrator import hydrate_documents


def _make_doc(path: str, **meta: object) -> Document:
    base = {
        "file_path": path,
        "file_hash": "h",
        "page": 1,
        "has_coordinates": True,
    }
    base.update(meta)
    return Document(page_content="x", metadata=base)


@pytest.mark.asyncio
async def test_citation_preserved_on_extract_failure(tmp_path: Path) -> None:
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), file_hash="h1", page=3)

    with (
        patch.object(
            document_hydrator.coord_cache, "get_coords_batch", return_value={}
        ),
        patch.object(
            document_hydrator, "_extract_pages_words_sync", return_value=[None]
        ),
    ):
        await hydrate_documents([doc])

    assert doc.metadata.get("coord_extract_failed") is True
    assert 3 in doc.metadata.get("citation_pages", [])


@pytest.mark.asyncio
async def test_full_page_extraction_ignores_chunk_bbox(tmp_path: Path) -> None:
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), file_hash="h2", page=1, bbox=[0, 0, 100, 100])

    # coords 테이블은 페이지 전체 단어 목록을 보관하므로, 캐시 오염 방지를 위해
    # 캐시 미스 시 청크 bbox를 무시하고 항상 전체 페이지(bbox=None)를 추출한다.
    def _extract(path: str, requests: list) -> list:
        assert requests == [(1, None)]
        return [[("a", 1.0, 2.0, 3.0, 4.0)]]

    with (
        patch.object(
            document_hydrator.coord_cache, "get_coords_batch", return_value={}
        ),
        patch.object(
            document_hydrator, "_extract_pages_words_sync", side_effect=_extract
        ),
    ):
        await hydrate_documents([doc])

    assert doc.metadata.get("word_coords") is not None


@pytest.mark.asyncio
async def test_already_hydrated_skipped(tmp_path: Path) -> None:
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(
        str(pdf),
        file_hash="h3",
        page=2,
        word_coords=[("a", 1.0, 2.0, 3.0, 4.0)],
    )

    spy_calls: list = []

    def _spy(path: str, requests: list) -> list:
        spy_calls.append((path, requests))
        return [[("a", 1.0, 2.0, 3.0, 4.0)] for _ in requests]

    with patch.object(document_hydrator, "_extract_pages_words_sync", side_effect=_spy):
        await hydrate_documents([doc])

    assert spy_calls == []
    assert doc.metadata["word_coords"] == [("a", 1.0, 2.0, 3.0, 4.0)]
