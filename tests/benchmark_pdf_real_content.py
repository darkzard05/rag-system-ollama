import time
import os
import sys
from pathlib import Path
import fitz
from typing import List

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from langchain_core.documents import Document
from core.rag_core import _load_pdf_docs


def create_large_real_pdf(source_path: str, target_path: str, multiplier: int):
    """실제 PDF를 여러 번 복제하여 대용량 실제 PDF 생성"""
    src_doc = fitz.open(source_path)
    out_doc = fitz.open()
    for _ in range(multiplier):
        out_doc.insert_pdf(src_doc)
    out_doc.save(target_path)
    out_doc.close()
    src_doc.close()


def _load_pdf_docs_sequential_real(file_path: str, file_name: str) -> List[Document]:
    docs = []
    with fitz.open(file_path) as doc:
        total_pages = len(doc)
        for i in range(total_pages):
            page = doc[i]
            text = page.get_text()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


def run_real_large_benchmark():
    src_pdf = "tests/2201.07520v1.pdf"
    if not os.path.exists(src_pdf):
        print(f"소스 PDF가 없습니다: {src_pdf}")
        return

    large_real_pdf = "tests/large_real_content.pdf"
    # 20페이지 논문을 5번 복제하여 100페이지로 생성
    print(f"실제 논문({src_pdf})을 복제하여 100페이지 대형 문서 생성 중...")
    create_large_real_pdf(src_pdf, large_real_pdf, 5)

    file_name = "large_real.pdf"

    try:
        # 1. 순차 방식 테스트
        print("\n[1] 순차 방식(Sequential) 실행 중...")
        start_time = time.time()
        docs_seq = _load_pdf_docs_sequential_real(large_real_pdf, file_name)
        seq_time = time.time() - start_time
        print(f"순차 방식 완료: {seq_time:.4f}초 (추출된 페이지: {len(docs_seq)})")

        # 2. 병렬 배치 방식 테스트 (최신 최적화 코드)
        print("\n[2] 병렬 배치 방식(Parallel Batch) 실행 중...")
        start_time = time.time()
        docs_para = _load_pdf_docs(large_real_pdf, file_name)
        para_time = time.time() - start_time
        print(f"병렬 방식 완료: {para_time:.4f}초 (추출된 페이지: {len(docs_para)})")

        # 결과 분석
        print("\n" + "=" * 60)
        print(f"{'항목':<25} | {'순차 방식':<15} | {'병렬 방식':<15}")
        print("-" * 60)
        print(f"{'소요 시간':<25} | {seq_time:<15.4f} | {para_time:<15.4f}")
        print(f"{'추출 페이지 수':<25} | {len(docs_seq):<15} | {len(docs_para):<15}")

        speedup = (seq_time / para_time) if para_time > 0 else 0
        print(f"\n성능 향상: {speedup:.2f}배 빨라짐")

        if speedup > 1.0:
            print(
                f"✅ 실제 복잡한 문서에서 {speedup:.2f}배의 속도 향상을 확인했습니다."
            )
        else:
            print("⚠️ 여전히 병렬화 효과가 작습니다. CPU 코어 할당량을 확인하세요.")
        print("=" * 60)

    finally:
        if os.path.exists(large_real_pdf):
            os.remove(large_real_pdf)


if __name__ == "__main__":
    run_real_large_benchmark()
