import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from core.document_processor import compute_file_hash, _detect_page_layout, load_pdf_docs

def test_compute_file_hash():
    """해시 계산 함수 검증"""
    data = b"test content"
    expected_hash = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
    assert compute_file_hash("dummy.txt", data=data) == expected_hash

def test_detect_page_layout_basic():
    """페이지 레이아웃 진단 로직 검증 (1단 구성 모사)"""
    mock_page = MagicMock()
    mock_page.rect.width = 600
    # (x0, y0, x1, y1, text, ...)
    mock_page.get_text.return_value = [
        (50, 50, 200, 70, "Header"),
        (50, 100, 500, 120, "Body text 1"),
        (50, 150, 500, 170, "Body text 2"),
    ]
    mock_page.get_drawings.return_value = [] # 선 없음
    
    result = _detect_page_layout(mock_page)
    
    assert result["is_multi_column"] is False
    assert result["strategy"] == "lines"

def test_detect_page_layout_multi_column():
    """2단 구성 레이아웃 진단 검증"""
    mock_page = MagicMock()
    mock_page.rect.width = 600
    # 왼쪽 단과 오른쪽 단에 분포된 텍스트 블록들
    blocks = []
    for i in range(10):
        blocks.append((50, 100 + i*20, 250, 115 + i*20, f"Left {i}"))
        blocks.append((400, 100 + i*20, 550, 115 + i*20, f"Right {i}")) # x0=400 (> 360)
    
    mock_page.get_text.return_value = blocks
    mock_page.get_drawings.return_value = []
    
    result = _detect_page_layout(mock_page)
    assert result["is_multi_column"] is True

@pytest.mark.asyncio
async def test_load_pdf_docs_filtering_logic():
    """TOC 및 참고문헌 필터링 로직 검증"""
    # fitz.open 및 pymupdf4llm.to_markdown 모킹
    with patch("fitz.open") as mock_fitz_open, \
         patch("pymupdf4llm.to_markdown") as mock_to_md, \
         patch("core.document_processor.compute_file_hash", return_value="hash123"), \
         patch("core.document_processor.SessionManager"):
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_doc.get_toc.return_value = [[1, "Introduction", 1], [1, "References", 9]]
        mock_fitz_open.return_value = mock_doc
        
        # 10페이지 분량의 가짜 마크다운 청크 생성
        mock_chunks = []
        for i in range(1, 11):
            text = f"Page {i} content"
            if i == 1: text = "Table of Contents\n1. Intro"
            if i == 9: text = "References\n[1] Paper A"
            
            mock_chunks.append({
                "text": text,
                "metadata": {"page": i, "page_count": 10},
                "words": []
            })
        mock_to_md.return_value = mock_chunks
        
        # Execute
        docs = load_pdf_docs("dummy.pdf", "dummy.pdf")
        
        # Verify
        pages = [d.metadata["page"] for d in docs]
        assert 1 in pages # TOC 포함 (현재 필터링 로직 제거됨)
        assert 9 in pages # References 포함
        assert 2 in pages # 본문 포함
