"""similarity 공용 헬퍼 동작 검증 (R: Group 6 중복 통합)."""

import numpy as np

from common.similarity import (
    cosine_distance,
    cosine_similarity,
    cosine_similarity_batch,
    normalize_vector,
)


def test_normalize_vector_single():
    v = np.array([3.0, 4.0])
    n = normalize_vector(v)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-9)


def test_normalize_vector_batch():
    v = np.array([[3.0, 4.0], [1.0, 0.0]])
    n = normalize_vector(v)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)


def test_normalize_eps_preserved():
    """1e-10 엡실론 정책이 유지되는지 (caching_optimizer 계약)."""
    v = np.zeros(3)
    n = normalize_vector(v, eps=1e-10)
    assert np.allclose(n, 0.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)


def test_cosine_similarity_same():
    # 정규화된 벡터 자신과의 유사도는 1.0 (cosine_similarity는 정규화 전제 내적)
    v = np.array([1.0, 2.0, 3.0])
    v_n = normalize_vector(v)
    assert np.isclose(cosine_similarity(v_n, v_n), 1.0)


def test_cosine_distance():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_distance(a, b), 1.0)


def test_cosine_similarity_batch_matches_manual():
    query = np.array([1.0, 1.0])
    docs = np.array([[1.0, 0.0], [1.0, 1.0]])
    result = cosine_similarity_batch(query, docs, eps=1e-9)
    # 수동 계산과 동등성
    q_n = query / (np.linalg.norm(query) + 1e-9)
    d_n = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
    expected = np.dot(d_n, q_n)
    assert np.allclose(result, expected, atol=1e-9)
