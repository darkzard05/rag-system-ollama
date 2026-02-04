import os
import sys
import time
from pathlib import Path

# src 디렉토리를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.utils import get_pdf_annotations
from langchain_core.documents import Document

def evaluate_highlighting():
    print("=== PDF 하이라이트 기능 성능 평가 ===")
    
    # 1. 테스트용 PDF 찾기
    possible_dirs = [
        ROOT_DIR / "tests" / "data",
        ROOT_DIR / "data" / "temp"
    ]
    
    pdf_files = []
    for d in possible_dirs:
        if d.exists():
            pdf_files.extend(list(d.glob("*.pdf")))
    
    if not pdf_files:
        print("❌ 테스트할 PDF 파일이 없습니다. tests/data 또는 data/temp에 PDF를 넣어주세요.")
        return

    # 2201.07520v1.pdf가 있으면 우선 선택
    target_pdf = next((f for f in pdf_files if "2201.07520" in f.name), pdf_files[0])
    test_pdf = str(target_pdf)
    print(f"대상 파일: {os.path.basename(test_pdf)}")

    # 2. 테스트용 가상 문서 조각 생성 (첫 페이지 텍스트 추출 시도)
    import fitz
    with fitz.open(test_pdf) as doc:
        page = doc[0]
        page_text = page.get_text()
        # 앞부분 100자를 검색 대상으로 삼음
        sample_text = page_text.strip()[:100]
        print(f"검색 텍스트 샘플: '{sample_text}...'")

    test_docs = [
        Document(page_content=sample_text, metadata={"page": 1})
    ]

    # 3. 속도 측정
    start_time = time.time()
    annotations = get_pdf_annotations(test_pdf, test_docs)
    duration = (time.time() - start_time) * 1000

    # 4. 결과 분석 및 정밀 검증
    print(f"\n--- 평가 결과 ---")
    print(f"소요 시간: {duration:.2f}ms")
    
    if annotations:
        print(f"추출 성공: {len(annotations)}개의 하이라이트 구역 발견")
        
        # [정밀 검증 1] 역추출 테스트 (Reverse Validation)
        # 추출된 좌표가 실제 텍스트를 포함하는지 확인
        print("\n--- 정밀 검증: 역추출 테스트 ---")
        try:
            with fitz.open(test_pdf) as doc:
                page = doc[0] # 1페이지라고 가정
                extracted_texts = []
                
                # 시각적 검증을 위한 드로잉 준비
                verify_img_path = str(ROOT_DIR / "reports" / "highlight_verification.png")
                os.makedirs(os.path.dirname(verify_img_path), exist_ok=True)
                
                # 좌표에 빨간 박스 그리기
                shape = page.new_shape()
                
                for i, anno in enumerate(annotations):
                    rect = fitz.Rect(anno['x'], anno['y'], anno['x'] + anno['width'], anno['y'] + anno['height'])
                    
                    # 1. 텍스트 역추출
                    clip_text = page.get_text("text", clip=rect).strip()
                    extracted_texts.append(clip_text)
                    print(f"  [{i+1}] 좌표 내 텍스트: '{clip_text}'")
                    
                    # 2. 박스 그리기
                    shape.draw_rect(rect)
                    
                shape.finish(color=(1, 0, 0), width=1.5) # 빨간색 테두리
                shape.commit()
                
                # 이미지 저장
                pix = page.get_pixmap()
                pix.save(verify_img_path)
                print(f"\n📸 검증 이미지 저장 완료: {verify_img_path}")
                
                # 3. 텍스트 일치율 확인
                full_extracted = " ".join(extracted_texts).replace("\n", " ")
                original_clean = sample_text.replace("\n", " ")
                
                # 간단한 포함 여부 확인
                match_ratio = 0
                if full_extracted:
                    # 공백 제거 후 비교
                    s1 = "".join(full_extracted.split())
                    s2 = "".join(original_clean.split())
                    
                    # 일부라도 겹치는지 확인 (PDF 추출 특성상 완벽 일치는 어려울 수 있음)
                    import difflib
                    matcher = difflib.SequenceMatcher(None, s1, s2)
                    match_ratio = matcher.ratio() * 100
                
                print(f"텍스트 일치율: {match_ratio:.1f}%")
                if match_ratio > 80:
                    print("✅ 좌표 정확도: 매우 높음 (텍스트가 정확히 포함됨)")
                elif match_ratio > 50:
                    print("⚠️ 좌표 정확도: 보통 (일부분만 포함되거나 노이즈 존재)")
                else:
                    print("❌ 좌표 정확도: 낮음 (엉뚱한 위치를 가리킴)")
                    
        except Exception as e:
            print(f"검증 중 오류 발생: {e}")

        # 점수 산정
        if duration < 100:
            speed_score = "매우 우수 (S)"
        elif duration < 300:
            speed_score = "우수 (A)"
        else:
            speed_score = "보통 (B)"
            
        print(f"\n속도 등급: {speed_score}")
    else:
        print("❌ 추출 실패: 텍스트 좌표를 찾지 못했습니다.")
        print("원인 분석: PDF 인코딩 문제이거나 텍스트 정규화 불일치 가능성")

if __name__ == "__main__":
    evaluate_highlighting()
