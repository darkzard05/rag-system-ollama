import time
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def benchmark_semantic_chunker():
    print("\n=== [1] Semantic Chunker: Breakpoint Detection ===")
    n = 10000
    similarities = np.random.rand(n)
    window_size = 3
    global_threshold = 0.5
    
    # [A] Current Logic (Loop-based)
    start = time.time()
    breakpoints_a = []
    weights = np.ones(window_size) / window_size
    local_avgs = np.convolve(similarities, weights, mode="valid")
    
    global_breaks_a = similarities < global_threshold
    for i in range(n):
        is_global_break = global_breaks_a[i]
        is_local_break = False
        if i >= window_size:
            avg = local_avgs[i - window_size]
            if similarities[i] < avg * 0.8:
                is_local_break = True
        if is_global_break or is_local_break:
            breakpoints_a.append(i + 1)
    end_a = time.time() - start
    
    # [B] Optimized Logic (Vectorized)
    start = time.time()
    global_breaks_b = similarities < global_threshold
    local_breaks_b = np.zeros_like(global_breaks_b)
    if n >= window_size:
        # similarities[i]와 local_avgs[i-window_size]를 비교 (i >= window_size)
        # 즉 similarities[window_size:] 와 local_avgs[:-1] 비교
        local_breaks_b[window_size:] = similarities[window_size:] < (local_avgs[:-1] * 0.8)
    
    # np.where를 사용하여 한 번에 인덱스 추출
    breakpoints_b = np.where(global_breaks_b | local_breaks_b)[0] + 1
    end_b = time.time() - start
    
    print(f"Current Loop: {end_a*1000:.4f}ms")
    print(f"Vectorized  : {end_b*1000:.4f}ms")
    print(f"Speedup     : {end_a/end_b:.2f}x")
    assert np.array_equal(breakpoints_a, breakpoints_b), "Results mismatch!"

def benchmark_reranker_diversity():
    print("\n=== [2] Reranker: Diversity Penalty (Bitset Matrix) ===")
    num_selected = 500
    bitset_dim = 256
    
    # 무작위 비트셋 생성
    res_bitset = np.random.choice([True, False], size=bitset_dim)
    selected_matrix = np.random.choice([True, False], size=(num_selected, bitset_dim))
    
    # [A] Current Logic (Loop-based)
    start = time.time()
    similarities_a = []
    for i in range(num_selected):
        sel_bitset = selected_matrix[i]
        intersection = np.logical_and(res_bitset, sel_bitset).sum()
        union = np.logical_or(res_bitset, sel_bitset).sum()
        similarities_a.append(intersection / union if union > 0 else 0.0)
    avg_sim_a = sum(similarities_a) / len(similarities_a)
    end_a = time.time() - start
    
    # [B] Optimized Logic (Matrix-based)
    start = time.time()
    # Matrix AND/OR 연산 (axis=1을 따라 합계 계산)
    intersections = np.logical_and(res_bitset, selected_matrix).sum(axis=1)
    unions = np.logical_or(res_bitset, selected_matrix).sum(axis=1)
    
    # 0으로 나누기 방지
    similarities_b = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions != 0)
    avg_sim_b = np.mean(similarities_b)
    end_b = time.time() - start
    
    print(f"Current Loop : {end_a*1000:.4f}ms")
    print(f"Matrix Ops   : {end_b*1000:.4f}ms")
    print(f"Speedup      : {end_a/end_b:.2f}x")
    assert np.isclose(avg_sim_a, avg_sim_b), f"Results mismatch! {avg_sim_a} vs {avg_sim_b}"

if __name__ == "__main__":
    benchmark_semantic_chunker()
    benchmark_reranker_diversity()