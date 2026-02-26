import time
import sys
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import numpy as np

# src 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from common.text_utils import bm25_tokenizer

def benchmark_bm25_rebuild():
    print("""
[Benchmark] BM25 리트리버 성능 비교 테스트
설명: 대규모 문서(1,000개 청크) 상황에서 질문 시마다 인덱스를 만드는 것과 
      캐싱된 것을 사용하는 것의 속도 차이를 측정합니다.
""")
    
    # 1. 가짜 데이터 생성 (1,000개 청크)
    print(f" - 데이터 생성 중: 1,000개의 청크 시뮬레이션...")
    dummy_docs = [
        Document(page_content=f"이것은 {i}번째 테스트 문서 청크입니다. 한국어 검색 성능을 확인합니다.", metadata={"page": i})
        for i in range(1000)
    ]

    # --- 기존 방식: 매번 재구축 ---
    print(" - [기존 방식] 인덱스 매번 재구축 시작...")
    start_rebuild = time.perf_counter()
    
    # 이 과정이 현재 aquery 호출 시마다 반복됨
    for _ in range(5): # 5번 반복 실행 평균 측정
        temp_retriever = BM25Retriever.from_documents(dummy_docs, preprocess_func=bm25_tokenizer)
        temp_retriever.k = 25
    
    end_rebuild = time.perf_counter()
    avg_rebuild = (end_rebuild - start_rebuild) / 5
    print(f"   => 소요 시간: {avg_rebuild:.4f}초 / query")

    # --- 개선 방식: 캐싱된 객체 활용 ---
    print("""
 - [개선 방식] 사전 구축된 인덱스 활용 시작...""")
    # 사전 구축 (retriever_manager 단계에서 1회 수행)
    cached_retriever = BM25Retriever.from_documents(dummy_docs, preprocess_func=bm25_tokenizer)
    
    start_cached = time.perf_counter()
    
    # 이 과정이 개선 후 aquery에서 수행될 내용
    for _ in range(5):
        # 객체는 풀에서 가져오고 k값만 변경 (가벼운 복사나 속성 변경)
        cached_retriever.k = 25
    
    end_cached = time.perf_counter()
    avg_cached = (end_cached - start_cached) / 5
    print(f"   => 소요 시간: {avg_cached:.8f}초 / query")

    # --- 결과 분석 ---
    improvement = avg_rebuild / avg_cached if avg_cached > 0 else 0
    print(f"""
[결과] 개선된 방식이 기존보다 약 {improvement:,.0f}배 빠릅니다.
{"*" * 50}""")
    if avg_rebuild > 0.1:
        print(f"경고: 현재 방식은 질문마다 {avg_rebuild:.2f}초의 불필요한 지연을 발생시키고 있습니다.")

if __name__ == "__main__":
    benchmark_bm25_rebuild()
