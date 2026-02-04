
import time
import numpy as np
import sys
import os
from dataclasses import dataclass, field

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.reranker import DiversityCalculator, RerankingResult

def old_calculate_content_similarity_set(content, selected_contents):
    if not selected_contents: return 0.0
    res_set = set(content)
    similarities = []
    for sel_content in selected_contents:
        sel_set = set(sel_content)
        common = len(res_set & sel_set)
        max_len = max(len(content), len(sel_content))
        similarities.append(common / max_len if max_len > 0 else 0.0)
    return sum(similarities) / len(similarities)

def run_reranker_benchmark():
    calc = DiversityCalculator()
    
    # 가상 데이터 생성: 100개의 검색 결과, 각 2000자 분량
    content = "This is a long sample document content for diversity testing. " * 50
    selected_contents = [f"This is selected doc {i} with some unique text. " + ("word " * 400) for i in range(100)]
    
    results = [RerankingResult(doc_id=f"doc_{i}", content=c, original_score=0.9, reranked_score=0.9, original_rank=i, final_rank=i) for i, c in enumerate(selected_contents)]
    target = RerankingResult(doc_id="target", content=content, original_score=0.9, reranked_score=0.9, original_rank=0, final_rank=0)

    print(f"Comparing 1 doc vs 100 docs (avg 2000 chars each)...")
    print("-" * 60)

    # 1. Old Method (Set-based)
    t0 = time.perf_counter()
    for _ in range(50):
        old_calculate_content_similarity_set(content, selected_contents)
    t_old = (time.perf_counter() - t0) / 50 * 1000

    # 2. New Method (Bitset-based)
    # Warm up cache
    calc.calculate_diversity_penalty(target, results)
    
    t1 = time.perf_counter()
    for _ in range(50):
        calc.calculate_diversity_penalty(target, results)
    t_new = (time.perf_counter() - t1) / 50 * 1000

    print(f"{'Method':<25} | {'Time (ms)':<12}")
    print("-" * 60)
    print(f"{'Old (Set-based)':<25} | {t_old:10.4f}")
    print(f"{'New (NumPy Bitset)':<25} | {t_new:10.4f}")
    print(f"Speedup: {t_old/t_new:8.1f}x")

if __name__ == "__main__":
    run_reranker_benchmark()
