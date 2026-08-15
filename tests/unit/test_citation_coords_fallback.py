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
        patch.object(document_hydrator, "_extract_page_words_sync", return_value=None),
    ):
        await hydrate_documents([doc])

    assert doc.metadata.get("coord_extract_failed") is True
    assert 3 in doc.metadata.get("citation_pages", [])


@pytest.mark.asyncio
async def test_full_page_retry_when_bbox_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"dummy")
    doc = _make_doc(str(pdf), file_hash="h2", page=1, bbox=[0, 0, 100, 100])

    def _extract(path: str, page_num: int, chunk_bbox: object) -> list | None:
        if chunk_bbox is not None:
            return None
        return [("a", 1.0, 2.0, 3.0, 4.0)]

    with (
        patch.object(
            document_hydrator.coord_cache, "get_coords_batch", return_value={}
        ),
        patch.object(
            document_hydrator, "_extract_page_words_sync", side_effect=_extract
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

    def _spy(path: str, page_num: int, chunk_bbox: object) -> list:
        spy_calls.append((path, page_num, chunk_bbox))
        return [("a", 1.0, 2.0, 3.0, 4.0)]

    with patch.object(document_hydrator, "_extract_page_words_sync", side_effect=_spy):
        await hydrate_documents([doc])

    assert spy_calls == []
    assert doc.metadata["word_coords"] == [("a", 1.0, 2.0, 3.0, 4.0)]
