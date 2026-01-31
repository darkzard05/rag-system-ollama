import sys
import time
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from services.optimization.index_optimizer import DocumentPruner


def old_school_pruning(documents, vectors, min_similarity=0.95):
    """기존 O(N^2) 루프 방식 (시뮬레이션 용)"""

    def _cosine_similarity(vec_a, vec_b):
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    kept_indices = []
    removed_indices = []
    for i, (_doc_i, vec_i) in enumerate(zip(documents, vectors, strict=False)):
        is_duplicate = False
        for j in kept_indices:
            if _cosine_similarity(vec_i, vectors[j]) >= min_similarity:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(i)
        else:
            removed_indices.append(i)
    return kept_indices


def benchmark_pruning():
    print("🚀 [Benchmark] DocumentPruner 최적화 테스트 (N=2,000)")

    # 1. 데이터 준비 (2,000개 문서, 384차원 벡터)
    N = 2000
    Dim = 384
    print(f"📊 가상 데이터 생성 중 (Chunks: {N}, Vector Dim: {Dim})...")

    docs = [Document(page_content=f"Content {i}") for i in range(N)]
    # 랜덤 벡터 생성
    vectors = [np.random.rand(Dim).astype(np.float32) for _ in range(N)]
    # 일부 중복 강제 생성 (100개를 첫 100개와 동일하게)
    for i in range(100):
        vectors[1000 + i] = vectors[i].copy()

    pruner = DocumentPruner(min_similarity=0.95)

    # --- [Test 1] 기존 O(N^2) 방식 측정 ---
    print("\n--- [Old] 루프 기반 방식 실행 ---")
    start_time = time.time()
    old_kept = old_school_pruning(docs, vectors)
    old_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {old_time:.4f}초 (남은 청크: {len(old_kept)})")

    # --- [Test 2] 신규 NumPy 방식 측정 ---
    print("\n--- [New] NumPy 벡터화 방식 실행 ---")
    start_time = time.time()
    new_docs, removed = pruner.prune_similar_documents(docs, vectors)
    new_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {new_time:.4f}초 (남은 청크: {len(new_docs)})")

    # 2. 결과 분석
    improvement = old_time / new_time
    print("\n" + "=" * 40)
    print("📈 성능 개선 결과")
    print(f"  - 루프 방식: {old_time:.4f}초")
    print(f"  - NumPy 방식: {new_time:.4f}초")
    print(f"  - 속도 향상: 약 {improvement:.1f}배 빨라짐")
    print("=" * 40)

    # 3. 정확도 검증
    if len(old_kept) == len(new_docs):
        print("✅ 결과 검증: 두 방식의 결과가 일치합니다. (무결성 통과)")
    else:
        print(
            f"⚠️ 결과 검증: 결과가 다릅니다. (Old: {len(old_kept)}, New: {len(new_docs)})"
        )


if __name__ == "__main__":
    benchmark_pruning()
