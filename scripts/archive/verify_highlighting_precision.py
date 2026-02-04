import os
import sys
import time
from pathlib import Path
import fitz
import random
import re

# src 디렉토리를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.utils import get_pdf_annotations
from langchain_core.documents import Document

def run_precision_test():
    print("=== [Precision Test] PDF 실제 텍스트 블록 역추적 검증 ===")
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_pdf):
        print("❌ 테스트 PDF 파일을 찾을 수 없습니다.")
        return

    # 1. PDF에서 랜덤하게 10개의 텍스트 블록 추출
    test_cases = []
    with fitz.open(test_pdf) as doc:
        # 1~5페이지 중 랜덤하게 선택
        for _ in range(10):
            p_idx = random.randint(0, min(4, len(doc)-1))
            page = doc[p_idx]
            blocks = page.get_text("blocks")
            # 텍스트가 있는 블록 중 글자수가 적당한 것(50~200자) 선택
            valid_blocks = [b for b in blocks if len(b[4].strip()) > 50 and len(b[4].strip()) < 300]
            if valid_blocks:
                b = random.choice(valid_blocks)
                test_cases.append({
                    "page": p_idx + 1,
                    "text": b[4].strip(),
                    "orig_rect": fitz.Rect(b[:4])
                })

    if not test_cases:
        print("❌ 테스트 케이스 추출 실패")
        return

    print(f"추출 완료: {len(test_cases)}개의 실제 문장으로 테스트 시작\n")

    results = []
    for i, case in enumerate(test_cases):
        print(f"[{i+1}/10] 테스트: P.{case['page']} | '{case['text'][:30]}...'")
        
        doc_obj = Document(page_content=case['text'], metadata={"page": case['page']})
        
        start_time = time.time()
        annotations = get_pdf_annotations(test_pdf, [doc_obj])
        duration = (time.time() - start_time) * 1000
        
        # 정밀도 검증 (IoU - Intersection over Union 비슷하게 구현)
        success = False
        if annotations:
            # 첫 번째 하이라이트가 원본 블록 좌표 근처에 있는지 확인
            anno = annotations[0]
            found_rect = fitz.Rect(anno['x'], anno['y'], anno['x'] + anno['width'], anno['y'] + anno['height'])
            orig = case['orig_rect']
            
            # 좌표가 겹치는지 확인 (조금이라도 겹치면 성공으로 간주)
            overlap = found_rect.intersect(orig)
            if overlap.width > 0 and overlap.height > 0:
                success = True

        results.append({"id": i+1, "success": success, "latency": duration})
        print(f"   결과: {'✅ PASS' if success else '❌ FAIL'} | {duration:.2f}ms")

    # 통계
    pass_count = sum(1 for r in results if r['success'])
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print("\n" + "="*50)
    print(f"📊 최종 정밀도 리포트: {pass_count}/10 ({pass_count*10}%)")
    print(f"⏱️ 평균 지연 시간: {avg_latency:.2f}ms")
    print("="*50)

if __name__ == "__main__":
    run_precision_test()
