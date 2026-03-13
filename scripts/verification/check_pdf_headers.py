import pymupdf4llm
import os

def check_actual_headers(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"파일 없음: {pdf_path}")
        return

    print(f"--- 실제 PDF 헤더 추출: {os.path.basename(pdf_path)} ---")
    
    # 마크다운으로 변환하여 헤더 패턴(#) 검색
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    lines = md_text.split('\n')
    headers = [line.strip() for line in lines if line.strip().startswith('#')]
    
    for h in headers[:20]: # 상위 20개만 확인
        print(h)

if __name__ == "__main__":
    check_actual_headers("tests/data/2201.07520v1.pdf")
