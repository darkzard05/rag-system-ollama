import numpy as np
import time

def legacy_calculate_similarities(embeddings: np.ndarray) -> list[float]:
    """현재 시스템에서 사용하는 방식 (비효율적)"""
    if len(embeddings) < 2:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
    similarities = np.sum(
        normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1
    )
    return similarities.tolist()

def optimized_calculate_similarities(embeddings: np.ndarray) -> list[float]:
    """np.einsum을 활용한 최적화 방식"""
    if len(embeddings) < 2:
        return []
    # 1. 정규화 최적화 (가드 로직 단순화)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms
    
    # 2. Einstein Summation을 사용한 내적 계산 (임시 배열 생성 없음)
    similarities = np.einsum('ij,ij->i', normalized_embeddings[:-1], normalized_embeddings[1:])
    return similarities.tolist()

def run_benchmark():
    # 시뮬레이션 데이터: 5,000개 문장, 384차원
    n_sentences = 5000
    dims = 384
    np.random.seed(42)
    data = np.random.randn(n_sentences, dims).astype(np.float32)
    
    print(f"--- 벤치마크 시작 (데이터: {n_sentences} 문장, {dims} 차원) ---")
    
    # Legacy 방식 측정
    start = time.time()
    for _ in range(100):
        res_legacy = legacy_calculate_similarities(data)
    dur_legacy = (time.time() - start) / 100
    print(f"[Legacy] 평균 소요 시간: {dur_legacy*1000:.4f} ms")
    
    # Optimized 방식 측정
    start = time.time()
    for _ in range(100):
        res_optimized = optimized_calculate_similarities(data)
    dur_optimized = (time.time() - start) / 100
    print(f"[Optimized] 평균 소요 시간: {dur_optimized*1000:.4f} ms")
    
    # 결과 일치 확인
    np.testing.assert_allclose(res_legacy, res_optimized, atol=1e-5)
    
    improvement = (dur_legacy - dur_optimized) / dur_legacy * 100
    print(f"\n성능 향상: {improvement:.2f}%")
    if improvement > 30:
        print("결론: 최적화 효과가 매우 뚜렷합니다. 수정을 권장합니다.")

if __name__ == "__main__":
    run_benchmark()
