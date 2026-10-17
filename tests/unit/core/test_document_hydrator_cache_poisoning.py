from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from core import document_hydrator
from core.document_hydrator import hydrate_documents


@pytest.mark.asyncio
async def test_hydrate_documents_cache_poisoning_prevention(tmp_path: Path) -> None:
    """Verify that hydrate_documents extracts full page (chunk_bbox=None) even for single chunk with bbox,
    ensuring page-level cache is not poisoned by partial bounding box extractions."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"dummy")

    doc1 = Document(
        page_content="상단 텍스트",
        metadata={
            "file_path": str(pdf),
            "file_hash": "test_hash",
            "page": 1,
            "pages": [1],
            "bbox": [0, 0, 100, 100],
            "has_coordinates": True,
        },
    )

    cache_hits: dict[int, list] = {}
    save_coords = AsyncMock(return_value=True)
    extracted_words = [("w1", 1, 1, 10, 10)]

    with (
        patch.object(
            document_hydrator.coord_cache,
            "get_coords_batch",
            return_value=cache_hits,
        ),
        patch.object(
            document_hydrator,
            "_extract_pages_words_sync",
            return_value=[extracted_words],
        ) as mock_extract,
        patch.object(
            document_hydrator.coord_cache,
            "save_coords",
            save_coords,
        ),
    ):
        await hydrate_documents([doc1])

    # Verify that _extract_pages_words_sync received chunk_bbox=None even though doc1 had a bbox
    mock_extract.assert_called_once()
    requests = mock_extract.call_args[0][1]
    assert requests == [(1, None)]

    # Verify save_coords was called with full page coords
    save_coords.assert_awaited_once_with("test_hash", 1, extracted_words)
