from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.document_processor import compute_file_hash, load_pdf_docs


def test_compute_file_hash():
    """해시 계산 함수 검증"""
    data = b"test content"
    expected_hash = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
    assert compute_file_hash("dummy.txt", data=data) == expected_hash


@pytest.mark.asyncio
async def test_load_pdf_docs_no_filtering():
    """모든 페이지가 필터링 없이 Document로 변환됩니다."""
    # fitz.open 및 pymupdf4llm.to_markdown 모킹
    with (
        patch("fitz.open") as mock_fitz_open,
        patch("pymupdf4llm.to_markdown") as mock_to_md,
        patch("core.document_processor.compute_file_hash", return_value="hash123"),
        patch("core.document_processor.SessionManager"),
    ):
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_doc.get_toc.return_value = [[1, "Introduction", 1], [1, "References", 9]]
        mock_fitz_open.return_value = mock_doc

        # 10페이지 분량의 가짜 마크다운 청크 생성
        mock_chunks = []
        for i in range(1, 11):
            text = f"Page {i} content"
            if i == 1:
                text = "Table of Contents\n1. Intro"
            if i == 9:
                text = "References\n[1] Paper A"

            mock_chunks.append(
                {"text": text, "metadata": {"page": i, "page_count": 10}, "words": []}
            )
        mock_to_md.return_value = mock_chunks

        # Execute
        docs = await load_pdf_docs("dummy.pdf", "dummy.pdf")

        # Verify
        pages = [d.metadata["page"] for d in docs]
        assert len(docs) == 10
        assert 1 in pages  # TOC 포함 (현재 필터링 로직 제거됨)
        assert 9 in pages  # References 포함
        assert 2 in pages  # 본문 포함
        assert all(isinstance(d, Document) for d in docs)
