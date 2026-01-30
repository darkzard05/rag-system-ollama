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

from core.rag_core import _load_pdf_docs, preprocess_text


# 비교를 위한 기존 순차 방식 구현
def _load_pdf_docs_sequential(file_path: str, file_name: str) -> list[Document]:
    docs = []
    with fitz.open(file_path) as doc_file:
        total_pages = len(doc_file)
        for page_num, page in enumerate(doc_file):
            text = page.get_text()
            if text:
                clean_text = preprocess_text(text)
                if clean_text and len(clean_text) > 10:
                    metadata = {
                        "source": file_name,
                        "page": int(page_num + 1),
                        "total_pages": int(total_pages),
                    }
                    docs.append(Document(page_content=clean_text, metadata=metadata))
    return docs


def run_benchmark():
    print("--- PDF 텍스트 추출 병렬화 성능 및 무결성 테스트 ---")

    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"테스트 파일이 없습니다: {pdf_path}")
        return

    file_name = "test_paper.pdf"

    # 1. 순차 방식 테스트
    print("\n[1] 순차 방식(Sequential) 실행 중...")
    start_time = time.time()
    docs_seq = _load_pdf_docs_sequential(pdf_path, file_name)
    seq_time = time.time() - start_time
    print(f"순차 방식 완료: {seq_time:.4f}초 (추출된 페이지: {len(docs_seq)})")

    # 2. 병렬 방식 테스트 (최적화된 코드 호출)
    print("\n[2] 병렬 방식(Parallel) 실행 중...")
    # SessionManager.add_status_log 등의 호출 시 에러 방지를 위해 간단한 Mock 처리 가능하나,
    # 현재 ThreadSafeSessionManager는 Streamlit 없이도 동작하므로 그대로 진행
    start_time = time.time()
    docs_para = _load_pdf_docs(pdf_path, file_name)
    para_time = time.time() - start_time
    print(f"병렬 방식 완료: {para_time:.4f}초 (추출된 페이지: {len(docs_para)})")

    # 결과 분석
    print("\n" + "=" * 60)
    print(f"{'검증 항목':<25} | {'순차 방식':<15} | {'병렬 방식':<15}")
    print("-" * 60)
    print(f"{'소요 시간':<25} | {seq_time:<15.4f} | {para_time:<15.4f}")
    print(f"{'추출 페이지 수':<25} | {len(docs_seq):<15} | {len(docs_para):<15}")

    # 무결성 검증
    print("-" * 60)

    # 1. 페이지 수 일치 확인
    page_count_match = len(docs_seq) == len(docs_para)

    # 2. 페이지 순서 확인
    order_correct = True
    for i, doc in enumerate(docs_para):
        if doc.metadata["page"] != i + 1:
            # 단, 원본에서 텍스트가 없는 페이지가 빠질 수 있으므로 실제 메타데이터 순서 확인
            if (
                i > 0
                and docs_para[i].metadata["page"] <= docs_para[i - 1].metadata["page"]
            ):
                order_correct = False
                break

    # 3. 내용 일치 확인 (첫 페이지와 마지막 페이지)
    content_match = False
    if page_count_match and len(docs_seq) > 0:
        content_match = (
            docs_seq[0].page_content == docs_para[0].page_content
            and docs_seq[-1].page_content == docs_para[-1].page_content
        )

    print(f"1. 페이지 수 일치 여부: {'✅ 일치' if page_count_match else '❌ 불일치'}")
    print(f"2. 페이지 순서 무결성: {'✅ 정상' if order_correct else '❌ 오류'}")
    print(f"3. 텍스트 내용 정확도: {'✅ 일치' if content_match else '❌ 불일치'}")

    speedup = (seq_time / para_time) if para_time > 0 else 0
    print(f"\n성능 향상: {speedup:.2f}배 빨라짐")
    print("=" * 60)

    if page_count_match and order_correct and content_match:
        print(
            "\n결론: PDF 병렬 추출 최적화가 무결성을 유지하며 성공적으로 적용되었습니다."
        )
    else:
        print("\n결론: 무결성 검증에 실패했습니다. 로직 확인이 필요합니다.")


if __name__ == "__main__":
    run_benchmark()
