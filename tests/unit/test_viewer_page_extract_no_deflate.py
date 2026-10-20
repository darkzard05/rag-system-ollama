"""단일 페이지 추출 시 deflate=False 최적화 및 유효성 검증 단위 테스트."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pymupdf

from ui.components.viewer import _get_pdf_page_bytes


def _create_sample_pdf(tmp_path: Path, pages: int = 5) -> str:
    path = str(tmp_path / "sample_perf.pdf")
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"PAGE {i + 1} TEST CONTENT TOKEN_XYZ")
    doc.save(path)
    doc.close()
    return path


def test_get_pdf_page_bytes_calls_tobytes_without_deflate(tmp_path: Path):
    """_get_pdf_page_bytes 내부에서 tobytes 호출 시 deflate=False로 고속 추출되는지 검증."""
    pdf_path = _create_sample_pdf(tmp_path, pages=3)

    real_tobytes = pymupdf.Document.tobytes
    tobytes_kwargs_list = []

    def spy_tobytes(self, *args, **kwargs):
        tobytes_kwargs_list.append(kwargs)
        return real_tobytes(self, *args, **kwargs)

    with patch.object(
        pymupdf.Document, "tobytes", side_effect=spy_tobytes, autospec=True
    ):
        page_bytes = _get_pdf_page_bytes(pdf_path, "hash_deflate_test", 1)

    assert len(tobytes_kwargs_list) == 1
    # deflate가 False로 전달되어 CPU 압축 오버헤드가 제거되었음을 단언
    assert tobytes_kwargs_list[0].get("deflate") is False
    assert tobytes_kwargs_list[0].get("garbage") == 1

    # 추출된 바이트가 여전히 유효한 단일 페이지 PDF인지 검증
    with pymupdf.open(stream=page_bytes, filetype="pdf") as doc:
        assert len(doc) == 1
        assert "TOKEN_XYZ" in doc[0].get_text("text")


def test_get_pdf_page_bytes_size_still_significantly_smaller(tmp_path: Path):
    """deflate=False 상태에서도 1페이지 추출본이 10페이지 전체 원본 파일 크기보다 충분히 작음을 검증."""
    pdf_path = _create_sample_pdf(tmp_path, pages=10)
    full_size = Path(pdf_path).stat().st_size

    page_bytes = _get_pdf_page_bytes(pdf_path, "hash_size_test", 1)

    # 10페이지 중 1페이지이므로 deflate를 하지 않더라도 전체 파일 대비 80% 미만이어야 함
    assert len(page_bytes) < full_size * 0.8
    assert len(page_bytes) > 0
