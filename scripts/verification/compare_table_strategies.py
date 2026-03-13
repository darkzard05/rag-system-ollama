import time
import os
import sys
import fitz
import pymupdf4llm

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))
from core.document_processor import _detect_page_layout

def compare_strategies():
    pdf_path = "tests/data/2201.07520v1.pdf"
    target_page_idx = 2 # Page 3 (0-indexed)
    
    doc = fitz.open(pdf_path)
    page = doc[target_page_idx]
    
    strategies = ["lines", "text", "adaptive"]
    results = {}

    print(f"\n{'='*80}")
    print(f" PDF Table Extraction Strategy Benchmark (Page {target_page_idx + 1})")
    print(f"{'='*80}\n")

    for strategy in strategies:
        print(f"[*] Running Strategy: '{strategy}'...")
        
        actual_strategy = strategy
        is_adaptive = False
        
        start_time = time.time()
        
        if strategy == "adaptive":
            is_adaptive = True
            layout = _detect_page_layout(page)
            actual_strategy = layout["strategy"]
            print(f"    [Adaptive Decision] Detected Strategy: {actual_strategy}")
        
        # 특정 페이지에 대해 추출 수행
        # (전체 문맥을 위해 doc을 전달하되 해당 페이지만 추출)
        md_text = pymupdf4llm.to_markdown(
            doc, 
            pages=[target_page_idx], 
            table_strategy=actual_strategy,
            graphics_limit=0 # 그래픽 노이즈 제거
        )
        
        duration = time.time() - start_time
        results[strategy] = {
            "time": duration,
            "content": md_text,
            "actual": actual_strategy
        }

    # 결과 비교 출력
    print("\n" + "="*80)
    print(f"{'Strategy':<15} | {'Actual':<10} | {'Latency':<10}")
    print("-" * 80)
    for s, data in results.items():
        print(f"{s:<15} | {data['actual']:<10} | {data['time']:.4f}s")
    print("="*80 + "\n")

    # 시각적 비교 (표 부분만 발췌)
    for s, data in results.items():
        print(f"--- [Output Preview: {s}] ---")
        lines = data["content"].split("\n")
        # 표 패턴(|)이 들어간 줄만 필터링해서 5줄 출력
        table_lines = [l for l in lines if "|" in l]
        if table_lines:
            for tl in table_lines[:6]:
                print(tl)
        else:
            print("(No structured table found in output)")
        print("-" * 40 + "\n")

    print("[ANALYSIS]")
    print("1. 'lines': 격자선이 없는 표에서는 데이터를 단순 텍스트로 인식하여 구조가 깨짐.")
    print("2. 'text': 선이 없어도 텍스트의 정렬 상태를 분석하여 표로 재구성 시도 (이 문서에 유리).")
    print("3. 'adaptive': 페이지를 사전 진단하여 최적의 전략을 선택하므로 성능과 품질의 균형을 맞춤.")

if __name__ == "__main__":
    compare_strategies()
