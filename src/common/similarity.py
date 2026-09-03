"""벡터 유사도/정규화 공용 유틸리티. (R: Group 6 중복 통합)

``async_reranker``, ``semantic_chunker_embeddings``, ``semantic_chunker_merge``,
``caching_optimizer`` 에 흩어져 있던 코사인 유사도/정규화 계산을 단일 진원으로
통합한다. 각 호출부의 **정규화 시점 / 엡실론 보호 정책은 그대로 보존**하도록
모든 헬퍼가 엡실론을 인자로 받는다 (동작 불변 원칙).
"""

from __future__ import annotations

import numpy as np


def normalize_vector(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """벡터(또는 배치 행렬)를 L2 정규화한다.

    - 1차원 벡터: ``v / (norm + eps)``
    - 2차원 배치: 각 행(row)을 ``axis=1`` 로 정규화
    - norm 이 0인 경우 그 자리 그대로 두고(0벡터 유지) ``eps`` 만큼만 보정한다.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm + eps)
    # 배치(2차원 이상): 행 단위 정규화, keepdims 로 브로드캐스트 맞춤
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """이미 (개별적으로) 정규화된 두 벡터의 코사인 유사도를 반환한다.

    호출자가 정규화를 보장하는 경우 사용한다 (``semantic_chunker_merge``,
    ``semantic_chunker_embeddings`` 처럼 사전 정규화된 벡터의 단순 내적).
    """
    return float(
        np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도를 1-유사도 거리로 변환한다 (chunker 인접 거리용)."""
    return 1.0 - cosine_similarity(a, b)


def cosine_similarity_batch(
    query_vec: np.ndarray, doc_vecs: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """쿼리 벡터와 문서 벡터 배치 간 코사인 유사도 벡터를 반환한다.

    재정렬은 reranker(사전 정규화 여부와 무관하게 항상 재정규화)만이 사용하므로,
    여기서는 쿼리와 배치를 모두 정규화한 뒤 내적한다 (eps 보호).
    """
    query_norm = normalize_vector(query_vec, eps)
    doc_norms = normalize_vector(doc_vecs, eps)
    return np.dot(doc_norms, query_norm)
