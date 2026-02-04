import pandas as pd

def simple_merge(list1, list2, k=10):
    """단순 병합 후 중복 제거"""
    seen = set()
    merged = []
    for doc in list1 + list2:
        if doc not in seen:
            merged.append(doc)
            seen.add(doc)
    return merged[:k]

def rrf_merge(list1, list2, k=10, constant=60):
    """RRF 알고리즘 적용"""
    rrf_scores = {}
    
    # 리스트 1 처리
    for rank, doc in enumerate(list1, start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + (1 / (constant + rank))
        
    # 리스트 2 처리
    for rank, doc in enumerate(list2, start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + (1 / (constant + rank))
        
    # 점수 순으로 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs[:k]]

def run_benchmark():
    # 시뮬레이션 데이터 (문서 ID)
    # BM25: 키워드 기반 우수 문서들
    bm25_results = [
        "Doc_A", # 키워드 1위
        "Doc_K",
        "Doc_C", # 키워드 3위
        "Doc_L",
        "Doc_M",
        "Doc_N",
        "Doc_O",
        "Doc_P",
        "Doc_Q",
        "Doc_R"
    ]
    
    # FAISS: 의미론적 기반 우수 문서들
    faiss_results = [
        "Doc_B", # 의미 1위
        "Doc_X",
        "Doc_C", # 의미 3위
        "Doc_Y",
        "Doc_Z",
        "Doc_A", # 의미 6위
        "Doc_S",
        "Doc_T",
        "Doc_U",
        "Doc_V"
    ]
    
    print("=== 하이브리드 검색 통합 알고리즘 비교 테스트 ===")
    print(f"BM25 상위권: {bm25_results[:3]}")
    print(f"FAISS 상위권: {faiss_results[:3]}")
    print("-" * 50)
    
    # 1. 단순 병합 결과
    simple_res = simple_merge(bm25_results, faiss_results, k=5)
    print(f"[Simple Merge] 결과: {simple_res}")
    # Simple Merge는 단순히 순서대로 합치므로 BM25 1위인 Doc_A가 무조건 1위가 됨
    
    # 2. RRF 결과
    rrf_res = rrf_merge(bm25_results, faiss_results, k=5)
    print(f"[RRF Merge   ] 결과: {rrf_res}")
    
    # 분석
    print("\n[분석 결과]")
    print(f"- Doc_C 순위: Simple({simple_res.index('Doc_C')+1 if 'Doc_C' in simple_res else 'N/A'}) vs RRF({rrf_res.index('Doc_C')+1 if 'Doc_C' in rrf_res else 'N/A'})")
    print(f"- Doc_A 순위: Simple({simple_res.index('Doc_A')+1 if 'Doc_A' in simple_res else 'N/A'}) vs RRF({rrf_res.index('Doc_A')+1 if 'Doc_A' in rrf_res else 'N/A'})")
    
    print("\n설명: Simple Merge는 먼저 합쳐진 리스트(BM25)의 순위에 편향되지만,")
    print("RRF는 양쪽 리스트에서 공통적으로 상위권인 'Doc_C'를 더 높게 평가하거나,")
    print("한쪽에서만 압도적인 것보다 양쪽의 균형을 고려하여 최적의 순위를 재배치합니다.")

if __name__ == "__main__":
    run_benchmark()
