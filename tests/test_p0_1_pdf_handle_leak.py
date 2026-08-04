"""
P0-1 결함 테스트: PDF 파일 핸들 누수

테스트 목표:
1. 정상적인 PDF 처리 확인
2. 파일 디스크립터 누수 감지
3. 메모리 안정성 (연속 처리)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psutil
import pytest

from core.document_processor import load_pdf_docs


def _create_test_pdf(tmp_path, name="test.pdf", pages=1):
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((50, 50), "Test content line\n" * 100)
    pdf_path = tmp_path / name
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.mark.asyncio
async def test_1_1_normal_pdf_load(tmp_path):
    """테스트 1.1: 정상적인 PDF 처리 확인"""
    test_pdf = _create_test_pdf(tmp_path)

    docs = await load_pdf_docs(test_pdf, "test.pdf")

    assert len(docs) > 0, "문서가 로드되지 않음"
    assert all(doc.page_content for doc in docs), "일부 문서에 컨텐츠 없음"


@pytest.mark.asyncio
async def test_1_2_file_descriptor_leak(tmp_path):
    """테스트 1.2: 핸들/리소스 누수 확인"""
    import gc

    process = psutil.Process(os.getpid())
    test_pdf = _create_test_pdf(tmp_path)

    gc.collect()
    initial_memory = process.memory_info().rss
    initial_fds = process.num_fds() if hasattr(process, "num_fds") else None  # type: ignore[attr-defined]

    # 100회 반복 처리
    for i in range(100):
        await load_pdf_docs(test_pdf, f"test_{i}.pdf")

    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)

    if initial_fds is not None:
        final_fds = process.num_fds()  # type: ignore[attr-defined]
        fd_increase = final_fds - initial_fds
        assert fd_increase <= 5, f"FD 누수 감지: {fd_increase}개 증가 (허용: 5개)"
    else:
        # Windows: 메모리 증가로 판단 (< 100MB)
        assert memory_increase_mb < 100, (
            f"메모리 누수 의심: {memory_increase_mb:.1f}MB 증가"
        )


@pytest.mark.asyncio
async def test_1_3_memory_stability(tmp_path):
    """테스트 1.3: 메모리 안정성 (연속 처리)"""
    import gc

    process = psutil.Process(os.getpid())
    test_pdf = _create_test_pdf(tmp_path, name="large_test.pdf", pages=5)

    gc.collect()
    initial_memory = process.memory_info().rss

    # 50회 연속 처리
    for i in range(50):
        await load_pdf_docs(test_pdf, f"large_{i}.pdf")

    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)

    # 검증: 메모리 증가 < 50MB
    assert memory_increase_mb < 50, f"메모리 누수 의심: {memory_increase_mb:.1f}MB 증가"
