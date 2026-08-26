"""C-Engine 폴백 경로의 병렬(스레드 풀) 페이지 추출 검증.

pymupdf4llm이 실패할 때(ONNXRuntimeError 등) 진입하는 Classic C-Engine 폴백 루프가
스레드 풀로 병렬화되었는지 확인합니다. 각 워커는 자체 PDF 핸들을 열고 닫으며
메인 ``doc``를 공유하지 않습니다(스레드 안전). 또한 순차 추출 결과와 동일한
페이지 순서/단어 좌표를 산출하는지 검증합니다.
"""

from unittest.mock import AsyncMock, patch

import fitz  # pymupdf
import pytest

from core.document_processor import load_pdf_docs, open_pdf_document


@pytest.mark.asyncio
async def test_cengine_fallback_parallel_matches_sequential(tmp_path):
    """병렬 C-Engine 추출이 순차 추출과 동일한 페이지/단어를 산출하고
    워커별 PDF 핸들을 사용하며 페이지 순서를 보존한다."""
    pdf_path = tmp_path / "doc.pdf"
    page_texts = [f"Page {i} unique content alpha{i}" for i in range(1, 7)]

    # 합성 PDF 생성 (페이지마다 고유 텍스트)
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(pdf_path))
    doc.close()

    # 기준(순차) 추출: 동일 파일을 fitz로 직접 추출
    ref = fitz.open(str(pdf_path))
    expected: dict[
        int, dict[str, str | list[tuple[float, float, float, float, str]]]
    ] = {}
    for idx in range(len(ref)):
        page = ref[idx]
        text_val = page.get_text("text").strip()
        words_val = [
            (float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]))
            for w in page.get_text("words")
        ]
        expected[idx + 1] = {"text": text_val, "words": words_val}
    ref.close()

    saved: dict[int, list] = {}
    open_calls = {"n": 0}

    def counting_open(fp: str):
        open_calls["n"] += 1
        return open_pdf_document(fp)

    with (
        patch(
            "pymupdf4llm.to_markdown",
            side_effect=RuntimeError("ONNXRuntimeError forced fallback"),
        ),
        patch("core.document_processor.open_pdf_document", side_effect=counting_open),
        patch("core.document_processor.compute_file_hash", return_value="testhash"),
        patch("core.document_processor.SessionManager"),
        patch("cache.coord_cache.coord_cache") as cc,
    ):

        async def fake_save(file_hash: str, page_num: int, words: list) -> None:
            saved[page_num] = words

        cc.save_coords = AsyncMock(side_effect=fake_save)

        docs = await load_pdf_docs(str(pdf_path), "doc.pdf")

    # 1) 페이지 순서 보존
    pages = [d.metadata["page"] for d in docs]
    assert pages == list(range(1, len(page_texts) + 1))

    # 2) 워커별 PDF 핸들 사용(공유 doc 없음): 최소 total_pages개의 핸들 오픈
    assert open_calls["n"] >= len(page_texts)

    # 3) 단어 좌표가 순차 기준과 페이지 순서로 일치, 튜플 형상 (x0,y0,x1,y1,text)
    for page_num, exp in expected.items():
        assert saved[page_num] == exp["words"]
        for word in saved[page_num]:
            assert len(word) == 5
            assert isinstance(word[4], str)
            assert all(isinstance(c, float) for c in word[:4])

    # 4) 텍스트 내용이 순차 추출과 일치
    texts = [d.page_content.strip() for d in docs]
    assert texts == [expected[i]["text"] for i in range(1, len(page_texts) + 1)]
