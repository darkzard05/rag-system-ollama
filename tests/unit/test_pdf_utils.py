"""pdf_utils 공용 헬퍼 동작 검증 (R: Group 7 중복 통합)."""

import pymupdf as fitz

from common.pdf_utils import get_pdf_page_count, open_pdf_document


def _make_pdf(tmp_path, pages: int) -> str:
    path = str(tmp_path / "sample.pdf")
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def test_open_pdf_document_yields_doc(tmp_path):
    path = _make_pdf(tmp_path, 2)
    with open_pdf_document(path) as doc:
        assert len(doc) == 2


def test_open_pdf_document_closes_handle(tmp_path):
    path = _make_pdf(tmp_path, 1)
    with open_pdf_document(path) as doc:
        doc_ref = doc
    # 컨텍스트 종료 후 열린 상태가 아니어야 함 (close 호출됨)
    assert doc_ref.is_closed


def test_get_pdf_page_count(tmp_path):
    path = _make_pdf(tmp_path, 3)
    assert get_pdf_page_count(path) == 3


def test_get_pdf_page_count_missing_file(tmp_path):
    assert get_pdf_page_count(str(tmp_path / "nope.pdf")) is None


def test_get_pdf_page_count_corrupt(tmp_path):
    path = str(tmp_path / "bad.pdf")
    with open(path, "wb") as f:
        f.write(b"not a real pdf")
    assert get_pdf_page_count(path) is None
