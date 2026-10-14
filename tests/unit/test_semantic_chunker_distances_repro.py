"""TDD Repro: EmbeddingBasedSemanticChunker._compute_distances vectorization & parity.

- Defect: _compute_distances uses a Python-level for loop over len(arr)-1 calling cosine_distance per pair.
- Contract: must be vectorized (e.g. numpy einsum) without per-pair scalar indexing, producing identical results (parity <= 1e-9).
"""

import numpy as np
import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class _ScalarIndexGuard(np.ndarray):
    """지정된 배열에 대해 정수(스칼라) 인덱싱(__getitem__) 시도시 예외를 발생시키는 가드 클래스."""

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            raise AssertionError("스칼라 인덱싱 발생: 비벡터화 루프 감지")
        return super().__getitem__(key)


def test_compute_distances_vectorization_and_parity() -> None:
    """_compute_distances가 스칼라 인덱싱 루프 없이 벡터화되어 동작하고 기존 루프와 수치 결과(parity)가 일치하는지 검증."""
    chunker = EmbeddingBasedSemanticChunker(embedder=None, buffer_size=1)  # type: ignore[arg-type]

    # 임의의 정규화된 임베딩 행렬 생성 (50청크, 32차원)
    np.random.seed(42)
    raw = np.random.rand(50, 32).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    # 1. 기존 루프(참조) 결과 계산
    ref_distances = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i].astype(np.float64)
        b = embeddings[i + 1].astype(np.float64)
        ref_distances.append(float(1.0 - np.dot(a, b)))

    # 2. _compute_distances 실행 (스칼라 인덱싱 가드 뷰 적용)
    guarded = embeddings.view(_ScalarIndexGuard)
    distances = chunker._compute_distances(guarded)

    assert len(distances) == len(ref_distances)
    assert np.allclose(distances, ref_distances, atol=1e-9)
