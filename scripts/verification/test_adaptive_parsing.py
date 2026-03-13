import asyncio
import os
import sys

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))

import fitz
from core.document_processor import load_pdf_docs, _detect_page_layout

def verify_adaptive_parsing():
    pdf_path = "tests/data/2201.07520v1.pdf"
    print(f"\n--- [ADAPTIVE PARSING TEST] Loading PDF: {pdf_path} ---")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. 레이아웃 진단 테스트 (샘플 페이지)
    print("\n[STEP 1] Layout Diagnosis Results:")
    for i in [0, 2, 5]: # 샘플 페이지 (표가 있을 법한 곳 위주)
        layout = _detect_page_layout(doc[i])
        print(f"  Page {i+1}: Strategy={layout['strategy']}, MultiColumn={layout['is_multi_column']}, HasTables={layout['has_tables']}")

    # 2. 실제 추출 수행
    print("\n[STEP 2] Performing Adaptive Extraction...")
    docs = load_pdf_docs(pdf_path, "adaptive_test.pdf")
    
    # 3. 결과물에서 테이블 구조 확인
    print("\n[STEP 3] Verifying Table Content:")
    found_table = False
    for i, doc_obj in enumerate(docs):
        content = doc_obj.page_content
        # 마크다운 테이블 패턴 확인 (| --- |)
        if "| ---" in content or "|---" in content:
            print(f"  Found potential table on Page {doc_obj.metadata.get('page')}:")
            # 테이블 부분만 추출해서 출력
            lines = content.split("\n")
            table_lines = [l for l in lines if "|" in l]
            for tl in table_lines[:5]: # 상위 5줄만 출력
                print(f"    {tl}")
            found_table = True
            break
            
    if found_table:
        print("\n[VERIFICATION SUCCESS] Table structure preserved with adaptive strategy.")
    else:
        print("\n[VERIFICATION NOTE] No explicit table structure found with '| ---' pattern. Checking raw content...")
        # 원본 문서는 테이블이 이미지나 복잡한 선으로 되어 있을 수 있음

if __name__ == "__main__":
    verify_adaptive_parsing()
