import os
import sys
import time
from pathlib import Path

import fitz

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from langchain_core.documents import Document

from core.rag_core import _extract_page_worker, _load_pdf_docs


def create_large_dummy_pdf(path: str, pages: int):
    """테스트를 위한 대용량 더미 PDF 생성"""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        # 각 페이지에 텍스트 삽입 (연산 부하를 위해 어느 정도 양의 텍스트 추가)
        text = f"이것은 {i + 1}페이지의 테스트용 텍스트입니다. " * 50
        page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()


def _load_pdf_docs_sequential_benchmark(
    file_path: str, file_name: str
) -> list[Document]:
    docs = []
    with fitz.open(file_path) as doc_file:
        total_pages = len(doc_file)
        for page_num in range(total_pages):
            # 실제 worker 함수를 사용하여 로직 동일하게 유지
            page_doc = _extract_page_worker(file_path, page_num, total_pages, file_name)
            if page_doc:
                docs.append(page_doc)
    return docs


def run_large_benchmark():
    print("--- 대용량 PDF(100p) 텍스트 추출 성능 테스트 ---")

    large_pdf_path = "tests/large_test_dummy.pdf"
    page_count = 100

    print(f"더미 PDF 생성 중 ({page_count}페이지)...")
    create_large_dummy_pdf(large_pdf_path, page_count)

    file_name = "large_dummy.pdf"

    try:
        # 1. 순차 방식 테스트
        print("\n[1] 순차 방식(Sequential) 실행 중...")
        start_time = time.time()
        docs_seq = _load_pdf_docs_sequential_benchmark(large_pdf_path, file_name)
        seq_time = time.time() - start_time
        print(f"순차 방식 완료: {seq_time:.4f}초")

        # 2. 병렬 방식 테스트
        print("\n[2] 병렬 방식(Parallel) 실행 중...")
        start_time = time.time()
        docs_para = _load_pdf_docs(large_pdf_path, file_name)
        para_time = time.time() - start_time
        print(f"병렬 방식 완료: {para_time:.4f}초")

        # 결과 분석
        print("\n" + "=" * 60)
        print(f"{'항목':<25} | {'순차 방식':<15} | {'병렬 방식':<15}")
        print("-" * 60)
        print(f"{'소요 시간':<25} | {seq_time:<15.4f} | {para_time:<15.4f}")
        print(f"{'추출 페이지 수':<25} | {len(docs_seq):<15} | {len(docs_para):<15}")

        speedup = (seq_time / para_time) if para_time > 0 else 0
        print(f"\n성능 향상: {speedup:.2f}배 빨라짐")

        if speedup > 1.2:
            print("✅ 대용량 문서에서 병렬화의 이점이 확실히 나타납니다.")
        else:
            print("⚠️ 성능 향상이 크지 않습니다. 시스템 환경을 확인하세요.")
        print("=" * 60)

    finally:
        if os.path.exists(large_pdf_path):
            os.remove(large_pdf_path)


if __name__ == "__main__":
    run_large_benchmark()
