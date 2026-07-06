# tests/stability/test_pdf_leak.py
import os
import psutil
import pytest
from src.core.document_processor import load_pdf_docs


def test_file_descriptor_leak_on_failure():
    process = psutil.Process(os.getpid())
    initial_fds = process.num_handles()

    # 존재하지 않거나 손상된 파일로 에러 유도 (10회 반복)
    for _ in range(10):
        try:
            # Note: path is relative to project root
            load_pdf_docs("non_existent_file_for_test.pdf", "test.pdf")
        except Exception:
            pass

    final_fds = process.num_handles()
    # 누수가 있다면 FD가 증가함.
    # 정상 상황이라면 FD 증가는 거의 없어야 함 (최대 시스템 버퍼 고려 2개 허용)
    assert final_fds <= initial_fds + 2, (
        f"FD leak detected: {initial_fds} -> {final_fds}"
    )
