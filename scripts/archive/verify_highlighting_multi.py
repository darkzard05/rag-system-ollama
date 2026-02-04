import os
import sys
import time
from pathlib import Path
import fitz
import re

# src 디렉토리를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.utils import get_pdf_annotations
from langchain_core.documents import Document

def run_multi_sentence_test():
    print("=== [Stress Test] 10개 문장 정밀 하이라이트 검증 ===")
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_pdf):
        print("❌ 테스트 PDF 파일을 찾을 수 없습니다.")
        return

    # 테스트할 10가지 다양한 유형의 문장 정의 (실제 PDF 내용 기반)
    test_cases = [
        {"page": 1, "text": "CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET"}, # 대제목
        {"page": 1, "text": "Multimodal models have demonstrated impressive results on a wide variety of tasks"}, # 본문 시작
        {"page": 1, "text": "recent models have moved to larger and more diverse datasets"}, # 중간 문장
        {"page": 2, "text": "The model is trained on a combination of structured and unstructured data"}, # 2페이지 본문
        {"page": 2, "text": "We evaluate CM3 on several zero-shot benchmarks"}, # 연구 방법론
        {"page": 3, "text": "Table 1 shows the performance comparison across different model sizes"}, # 표 참조 문장
        {"page": 3, "text": "CM3-Medium achieves competitive results with only a fraction of the parameters"}, # 성능 강조
        {"page": 4, "text": "The attention mechanism allows for efficient cross-modal information exchange"}, # 기술적 세부사항
        {"page": 5, "text": "We conclude that causal masking is a powerful objective for multimodal pre-training"}, # 결론
        {"page": 5, "text": "Future work will explore scaling CM3 to even larger datasets"} # 향후 계획
    ]

    results = []
    total_start = time.time()

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/10] 테스트 중: '{case['text'][:40]}...' (Page {case['page']})")
        
        doc_obj = Document(page_content=case['text'], metadata={"page": case['page']})
        
        start_time = time.time()
        annotations = get_pdf_annotations(test_pdf, [doc_obj])
        duration = (time.time() - start_time) * 1000
        
        # 정밀도 검증 (역추출)
        match_quality = "FAIL"
        extracted_text = ""
        if annotations:
            try:
                with fitz.open(test_pdf) as doc:
                    page = doc[case['page']-1]
                    parts = []
                    for anno in annotations:
                        rect = fitz.Rect(anno['x'], anno['y'], anno['x'] + anno['width'], anno['y'] + anno['height'])
                        parts.append(page.get_text("text", clip=rect).strip())
                    extracted_text = " ".join(parts).replace("\n", " ")
                    
                    # 유사도 체크 (공백 제거 후 비교)
                    s1 = re.sub(r'\s+', '', extracted_text.lower())
                    s2 = re.sub(r'\s+', '', case['text'].lower())
                    
                    if s2 in s1 or s1 in s2 or len(set(s1) & set(s2)) / max(len(s1), len(s2)) > 0.7:
                        match_quality = "PASS"
            except Exception as e:
                match_quality = f"ERROR ({e})"

        results.append({
            "id": i+1,
            "match": match_quality,
            "latency": duration,
            "extracted": extracted_text[:50] + "..." if extracted_text else "N/A"
        })
        print(f"   결과: {match_quality} | 소요시간: {duration:.2f}ms")

    total_duration = time.time() - total_start
    
    # 요약 보고서
    print("\n" + "="*50)
    print("📊 최종 테스트 요약 보고서")
    print("="*50)
    success_count = sum(1 for r in results if r['match'] == "PASS")
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"✅ 최종 성공률: {success_count}/10 ({success_count*10:.1f}%)")
    print(f"⏱️ 평균 지연 시간: {avg_latency:.2f}ms")
    print(f"⌛ 총 소요 시간: {total_duration:.2f}s")
    print("-" * 50)
    
    if success_count >= 8:
        print("결과 판정: 🟢 매우 우수 (운영 환경 적용 적합)")
    elif success_count >= 6:
        print("결과 판정: 🟡 양호 (일부 복잡한 레이아웃 보완 필요)")
    else:
        print("결과 판정: 🔴 미흡 (엔진 로직 재검토 필요)")

if __name__ == "__main__":
    run_multi_sentence_test()
