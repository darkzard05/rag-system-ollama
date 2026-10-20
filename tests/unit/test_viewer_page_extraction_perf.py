from __future__ import annotations

from pathlib import Path

import pymupdf

from ui.components.viewer import _get_pdf_page_bytes

_TOKEN = "TEST_FAST_EXTRACT_TOKEN_999"


def _create_multi_page_pdf(tmp_path: Path, pages: int = 10) -> str:
    pdf_path = str(tmp_path / "multi_test.pdf")
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"PAGE {i + 1} BODY TEXT")
        if i == 0:
            page.insert_text((72, 200), f"TARGET: {_TOKEN}")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_insert_pdf_extracts_exact_single_page_and_token(tmp_path: Path):
    """_get_pdf_page_bytes가 insert_pdf 방식으로 정확히 1페이지만 반환하고 토큰을 보존하는지 검증."""
    pdf_path = _create_multi_page_pdf(tmp_path, pages=5)

    page_bytes = _get_pdf_page_bytes(pdf_path, "test_hash_fast", 1)

    assert page_bytes != b""
    with pymupdf.open(stream=page_bytes, filetype="pdf") as single_doc:
        assert len(single_doc) == 1
        found = single_doc[0].search_for(_TOKEN)
        assert len(found) == 1


def test_insert_pdf_out_of_range_and_invalid(tmp_path: Path):
    """유효하지 않은 페이지 요청 시 예외 없이 빈 바이트 반환 검증."""
    pdf_path = _create_multi_page_pdf(tmp_path, pages=3)

    assert _get_pdf_page_bytes(pdf_path, "test_hash_fast", 0) == b""
    assert _get_pdf_page_bytes(pdf_path, "test_hash_fast", 4) == b""
    assert (
        _get_pdf_page_bytes(str(tmp_path / "non_existent.pdf"), "test_hash_fast", 1)
        == b""
    )


def test_insert_pdf_shrinks_payload_size(tmp_path: Path):
    """추출된 단일 페이지가 원본보다 유의미하게 작음을 검증 (기존 캐시 축소 계약 유지)."""
    pdf_path = _create_multi_page_pdf(tmp_path, pages=10)
    full_size = Path(pdf_path).stat().st_size

    page_bytes = _get_pdf_page_bytes(pdf_path, "test_hash_fast", 1)
    assert len(page_bytes) < full_size * 0.8
