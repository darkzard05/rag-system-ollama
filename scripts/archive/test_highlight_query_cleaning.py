import fitz  # PyMuPDF
import re
import os

# 테스트용 PDF 경로
PDF_PATH = "tests/data/2201.07520v1.pdf"

def get_pdf_text(pdf_path, page_num=0):
    """PDF의 특정 페이지 텍스트를 추출합니다."""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return ""
    
    try:
        doc = fitz.open(pdf_path)
        if page_num < len(doc):
            page = doc[page_num]
            return page.get_text()
        else:
            print(f"Error: Page {page_num} does not exist.")
            return ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def current_logic(content):
    """현재 src/common/utils.py에 있는 로직"""
    # [개선] 마크다운 특수문자(#, *, ` 등)를 제거하여 실제 PDF 텍스트와 매칭율 향상
    clean_content = re.sub(r"[#*`_~\[\]()]", "", content).lower()

    # [최적화] 청크 전체를 하이라이트하기 위해 모든 문장을 검색 대상으로 설정
    sentences = [
        s.strip()
        for s in re.split(r"[.!?\n]", clean_content)
        if len(s.strip()) > 8  # 너무 짧은 검색어는 무시
    ]
    return sentences

def improved_logic(content):
    """개선된 검색 쿼리 전처리 로직"""
    # 1. HTML 태그 제거 (예: <img src="...">)
    text = re.sub(r'<[^>]+>', ' ', content)
    
    # 2. 마크다운 및 특수문자 제거 (기존보다 강화, 따옴표 포함)
    text = re.sub(r"[#*`_~\[\]()\"']", "", text)
    
    # 3. 연속 공백 제거 및 앞뒤 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. 소문자 변환
    text = text.lower()

    # 5. 문장 분리 (줄바꿈 포함)
    raw_sentences = re.split(r"[.!?\n]", text)
    
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        
        # [필터링 1] 최소 길이 상향 (8 -> 20)
        # 너무 짧은 문장은 오탐지(False Positive)의 원인이 됨
        if len(s) < 20:
            continue
            
        # [필터링 2] 숫자나 특수문자로만 구성된 쓰레기 데이터 제거
        if re.match(r'^[\d\s\W]+$', s):
            continue
            
        # [필터링 3] 표/그림 캡션 등 불필요한 메타데이터 제거 (휴리스틱)
        # 예: "table 1", "figure 3", "page 10" 등으로 시작하는 경우
        if re.match(r'^(table|figure|fig\.|tab\.)\s*\d+', s):
            continue
            
        # [필터링 4] 참고문헌 패턴 (예: [1], (2020)) 등으로 시작하는 경우
        if re.match(r'^[\(\[]\s*\d+\s*[\)\]]', s):
            continue

        sentences.append(s)
        
    return sentences

def main():
    print(f"Testing with file: {PDF_PATH}")
    
    # 1, 2, 5 페이지 텍스트 추출 (로그에서 문제가 되었던 페이지들)
    target_pages = [1, 4] # 0-indexed (Page 2, Page 5)
    
    for page_idx in target_pages:
        print(f"\n{'='*20} Page {page_idx + 1} {'='*20}")
        content = get_pdf_text(PDF_PATH, page_idx)
        if not content:
            continue
            
        print(f"--- Original Text Preview (First 100 chars) ---")
        print(content[:100].replace('\n', ' '))
        print("-" * 50)

        # 현재 로직 실행
        current_queries = current_logic(content)
        print(f"\n[Current Logic] Generated {len(current_queries)} queries:")
        for q in current_queries[:5]: # 5개만 출력
            print(f"  - '{q}'")
        if len(current_queries) > 5: print("  ... (rest omitted)")

        # 개선된 로직 실행
        improved_queries = improved_logic(content)
        print(f"\n[Improved Logic] Generated {len(improved_queries)} queries:")
        for q in improved_queries[:5]: # 5개만 출력
            print(f"  - '{q}'")
        if len(improved_queries) > 5: print("  ... (rest omitted)")
        
        # 제거된 쿼리 확인 (Current에는 있는데 Improved에는 없는 것)
        removed = set(current_queries) - set(improved_queries)
        if removed:
            print(f"\n[Filtered Out] Queries removed by new logic ({len(removed)}):")
            # 길이가 짧은 순서대로 정렬하여 어떤 '노이즈'가 제거되었는지 확인
            sorted_removed = sorted(list(removed), key=len)
            for q in sorted_removed[:10]:
                print(f"  - (Len: {len(q)}) '{q}'")
            if len(sorted_removed) > 10: print("  ... (rest omitted)")

if __name__ == "__main__":
    main()
