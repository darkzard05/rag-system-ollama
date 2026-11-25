import os
import fitz  # PyMuPDF
import logging
from core.document_processor import load_pdf_docs
from common.utils import extract_annotations_from_docs

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_highlight_accuracy(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"\n--- 하이라이트 정확도 평가 시작: {os.path.basename(pdf_path)} ---")
    
    # 1. 문서 로드 및 좌표 추출
    docs = load_pdf_docs(pdf_path, os.path.basename(pdf_path))
    
    # 테스트를 위해 상위 3개 청크만 샘플링
    sample_docs = docs[:3]
    
    # 2. 하이라이트(Annotations) 생성
    annotations = extract_annotations_from_docs(sample_docs)
    
    # 3. 실제 PDF와 대조 검증
    doc = fitz.open(pdf_path)
    
    total_hits = 0
    total_misses = 0
    
    for i, ann in enumerate(annotations):
        page_idx = ann["page"]
        if page_idx >= len(doc):
            print(f"[{i+1}] 페이지 번호 초과: {page_idx}")
            continue
            
        page = doc[page_idx]
        
        # 주석 좌표 (x, y, width, height) -> fitz.Rect (x0, y0, x1, y1)
        rect = fitz.Rect(ann["x"], ann["y"], ann["x"] + ann["width"], ann["y"] + ann["height"])
        
        # 해당 좌표에서 텍스트 직접 재추출
        extracted_text = page.get_text("text", clip=rect).strip().replace("\n", " ")
        expected_text = sample_docs[i].page_content.strip().replace("\n", " ")
        
        # 일치도 계산: 추출된 텍스트 조각들이 원문에 포함되어 있는지 확인
        test_words = [w for w in extracted_text.split() if len(w) > 2]
        is_hit = False
        if test_words:
            match_count = sum(1 for w in test_words if w.lower() in expected_text.lower())
            is_hit = match_count / len(test_words) > 0.5 # 50% 이상의 단어가 겹치면 적중
        
        if is_hit:
            total_hits += 1
            status = "✅ 적중"
        else:
            total_misses += 1
            status = "❌ 불일치"
            
        print(f"[{i+1}] 페이지 {page_idx+1}: {status}")
        print(f"    - 예상: {expected_text[:60]}...")
        print(f"    - 추출: {extracted_text[:60]}...")
        print("-" * 30)

    # 4. 결과 리포트
    accuracy = (total_hits / len(annotations)) * 100 if annotations else 0
    print(f"\n📊 최종 평가 결과")
    print(f"  - 총 검증 영역: {len(annotations)}")
    print(f"  - 적중: {total_hits}")
    print(f"  - 미적중: {total_misses}")
    print(f"  - 정확도: {accuracy:.1f}%")

if __name__ == "__main__":
    import glob
    # 테스트용 PDF 경로 수정 (실제 프로젝트 구조에 맞춤)
    test_pdfs = glob.glob("tests/data/*.pdf")
    if not test_pdfs:
        # data 폴더도 확인
        test_pdfs = glob.glob("data/**/*.pdf", recursive=True)
        
    if test_pdfs:
        evaluate_highlight_accuracy(test_pdfs[0])
    else:
        print("테스트용 PDF 파일을 찾을 수 없습니다.")
