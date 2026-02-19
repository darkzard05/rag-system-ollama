import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.document_processor import load_pdf_docs

def test_extraction():
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)
    
    print(f"--- Extraction Test Started: {file_name} ---")
    
    try:
        # PyMuPDF4LLM 엔진을 사용한 로딩
        docs = load_pdf_docs(test_pdf, file_name)
        
        print(f"Successfully extracted {len(docs)} pages.")
        
        if docs:
            first_doc = docs[0]
            print("\\n[Page 1 Metadata]")
            for k, v in first_doc.metadata.items():
                print(f"  {k}: {v}")
            
            print("\\n[Page 1 Content Sample (First 500 chars)]")
            print("-" * 50)
            print(first_doc.page_content[:500])
            print("-" * 50)
            
            if len(docs) > 1:
                last_doc = docs[-1]
                print(f"\\n[Last Page ({len(docs)}) Content Sample]")
                print("-" * 50)
                print(last_doc.page_content[:300])
                print("-" * 50)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extraction()
