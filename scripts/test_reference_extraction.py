
import os
import sys
import logging

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from core.document_processor import load_pdf_docs
from core.session import SessionManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def test_reference_extraction():
    # 1. 테스트용 PDF 파일 경로
    test_pdf = "tests/data/2201.07520v1.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"테스트 파일 {test_pdf}가 없습니다.")
        return

    print(f"--- {test_pdf} 분석 테스트 시작 ---")
    
    # 2. 문서 처리 실행
    docs = load_pdf_docs(test_pdf, "test_ref_sample.pdf")
    
    print(f"\n[DEBUG] 추출된 Document 수: {len(docs)}")
    
    # 첫 번째 문서의 메타데이터 샘플 출력
    if docs:
        print(f"[DEBUG] 첫 번째 문서 메타데이터 키: {docs[0].metadata.keys()}")
    
    # 3. 결과 검증
    ref_found = False
    for i, doc in enumerate(docs):
        linked_refs = doc.metadata.get("linked_references")
        if linked_refs:
            ref_found = True
            page = doc.metadata.get("page")
            print(f"[SUCCESS] {page}페이지에서 {len(linked_refs)}개의 레퍼런스 연결 발견:")
            for num, ref in linked_refs.items():
                print(f"  - {num}: {ref[:60]}...")
                
    if not ref_found:
        print("\n[DEBUG] 레퍼런스 연결 실패. 원인 파악을 위해 파싱된 텍스트 일부를 확인합니다.")
        if docs:
            print(f"본문 샘플 (1페이지): {docs[0].page_content[:300]}")
    else:
        print("\n[FINAL] 레퍼런스 추출 및 본문 매핑 기능이 정상적으로 작동합니다.")

if __name__ == "__main__":
    # 세션 초기화 (필요 시)
    test_reference_extraction()
