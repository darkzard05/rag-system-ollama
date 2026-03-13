import asyncio
import os
import json
from core.document_processor import load_pdf_docs
from langchain_core.documents import Document

async def verify_pdf_extraction():
    # 테스트 파일 경로 (논문 PDF)
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Test PDF not found at {pdf_path}")
        return

    print(f"\n--- Testing Extraction on: {os.path.basename(pdf_path)} ---")
    
    # 1. 문서 로드 실행
    try:
        docs = load_pdf_docs(pdf_path, os.path.basename(pdf_path))
        print(f"✅ Successfully loaded {len(docs)} pages.")
        
        # 2. 첫 페이지 및 섹션 정보 확인
        for i, doc in enumerate(docs[:5]):
            print(f"\n[Page {doc.metadata.get('page')}]")
            print(f"- Content length: {len(doc.page_content)}")
            print(f"- Current Section: {doc.metadata.get('current_section')}")
            print(f"- Has Tables: {doc.metadata.get('has_tables')}")
            print(f"- Table Count: {doc.metadata.get('table_count', 0)}")
            
            # 표 내용 일부 출력 (마크다운 표 확인)
            if doc.metadata.get("has_tables"):
                print("- Table Preview (first 100 chars):")
                # 마크다운 표 패턴 (|---|) 검색
                import re
                table_match = re.search(r"\|.*\|", doc.page_content)
                if table_match:
                    start = table_match.start()
                    print(doc.page_content[start:start+200] + "...")
                else:
                    print("  (No markdown table pattern found in text content)")

        # 3. 메타데이터 무결성 확인
        print("\n--- Metadata Check ---")
        sample_meta = docs[0].metadata
        required_keys = ["source", "page", "total_pages", "has_coordinates", "has_tables"]
        for key in required_keys:
            status = "✅" if key in sample_meta else "❌"
            print(f"{status} {key}: {sample_meta.get(key)}")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_pdf_extraction())
