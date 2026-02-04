
import time
import numpy as np
import sys
import os
from dataclasses import dataclass

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.search_aggregator import SearchResultAggregator, AggregationStrategy

@dataclass
class MockResult:
    doc_id: str
    content: str
    score: float
    node_id: str
    metadata: dict = None

def run_aggregator_benchmark():
    aggregator = SearchResultAggregator()
    
    # 가상 데이터 생성: 4개 노드, 각 500개 결과 (총 2,000개)
    num_nodes = 4
    results_per_node = 500
    search_results = {}
    
    for n in range(num_nodes):
        node_id = f"node_{n}"
        results = []
        for i in range(results_per_node):
            results.append(MockResult(
                doc_id=f"doc_{i}", # 중복 발생 유도
                content=f"Sample content for doc {i}",
                score=np.random.rand(),
                node_id=node_id
            ))
        search_results[node_id] = results

    print(f"Aggregating {num_nodes * results_per_node} results from {num_nodes} nodes...")
    print("-" * 60)

    # 1. Weighted Score (Existing)
    t0 = time.perf_counter()
    for _ in range(10):
        aggregator.aggregate_results(search_results, strategy=AggregationStrategy.WEIGHTED_SCORE)
    t_weighted = (time.perf_counter() - t0) / 10 * 1000

    # 2. RRF Fusion (New Optimized)
    t1 = time.perf_counter()
    for _ in range(10):
        aggregator.aggregate_results(search_results, strategy=AggregationStrategy.RRF_FUSION)
    t_rrf = (time.perf_counter() - t1) / 10 * 1000

    print(f"{'Strategy':<20} | {'Time (ms)':<12}")
    print("-" * 60)
    print(f"{'Weighted Score':<20} | {t_weighted:10.4f}")
    print(f"{'RRF (NumPy)':<20} | {t_rrf:10.4f}")
    print(f"RRF Overhead Ratio: {t_rrf/t_weighted:.2f}x (Higher quality but slightly more complex)")

if __name__ == "__main__":
    run_aggregator_benchmark()
