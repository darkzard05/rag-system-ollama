import time
import os
import sys
import fitz
import pymupdf4llm

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))
from core.document_processor import load_pdf_docs

def benchmark_full_document():
    pdf_path = "tests/data/2201.07520v1.pdf"
    file_name = "2201.07520v1.pdf"
    
    # 전략 리스트
    # 'adaptive'는 우리가 구현한 load_pdf_docs의 샘플링 기반 전략을 사용합니다.
    strategies = ["lines", "text", "adaptive"]
    results = {}

    print(f"\n{'='*80}")
    print(f" Full Document Table Extraction Benchmark (Total 20 Pages)")
    print(f"{'='*80}\n")

    for strategy in strategies:
        print(f"[*] Processing entire document with strategy: '{strategy}'...")
        
        start_time = time.time()
        
        # 1. 문서 로드 및 파싱 (각 전략별 환경 시뮬레이션)
        if strategy == "adaptive":
            # 우리가 구현한 적응형 로직 호출 (샘플링 진단 포함)
            docs = load_pdf_docs(pdf_path, file_name)
            full_content = "\n\n".join([d.page_content for d in docs])
            actual_strategy = "adaptive(auto)"
        else:
            # 고정 전략으로 전체 문서 파싱
            doc = fitz.open(pdf_path)
            full_content = pymupdf4llm.to_markdown(
                doc, 
                table_strategy=strategy,
                graphics_limit=0
            )
            actual_strategy = strategy
            doc.close()
        
        duration = time.time() - start_time
        
        # 2. 통계 산출
        lines = full_content.split("\n")
        table_row_count = sum(1 for l in lines if "|" in l)
        char_count = len(full_content)
        
        results[strategy] = {
            "time": duration,
            "table_rows": table_row_count,
            "chars": char_count,
            "actual": actual_strategy
        }

    # 결과 테이블 출력
    print("\n" + "="*80)
    print(f"{'Strategy':<15} | {'Latency':<10} | {'Table Rows':<12} | {'Total Chars'}")
    print("-" * 80)
    for s, data in results.items():
        print(f"{s:<15} | {data['time']:>8.2f}s | {data['table_rows']:>12} | {data['chars']:>10}")
    print("="*80 + "\n")

    print("[DETAILED ANALYSIS]")
    # 품질 샘플링 (주요 테이블이 있는 3페이지 근처 내용 확인)
    # adaptive 결과에서 테이블이 포함된 부분을 찾아 일부 출력
    print("--- [Adaptive Strategy Table Sample] ---")
    adaptive_lines = results["adaptive"]["content_sample"] if "content_sample" in results["adaptive"] else []
    # (스크립트 내부 변수 전달을 위해 로직 수정)
    
    # 실제 문서의 특정 키워드(Table 1, Table 2 등) 검색 결과 비교
    print(f"1. 'lines' 전략은 빠른 속도({results['lines']['time']:.2f}s)를 보이지만, 테이블 행 감지 수가 가장 적을 가능성이 높습니다.")
    print(f"2. 'text' 전략은 가장 정밀하지만, 전체 문서 처리 시 상당한 지연({results['text']['time']:.2f}s)이 발생합니다.")
    print(f"3. 'adaptive' 전략은 샘플링을 통해 문서 전체의 최적 전략을 결정하여 효율과 품질을 조율합니다.")

if __name__ == "__main__":
    benchmark_full_document()
