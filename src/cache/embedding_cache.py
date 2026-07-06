"""
텍스트 청크별 임베딩 벡터를 디스크에 캐싱하여 중복 계산을 방지하는 모듈.
"""

import hashlib
import logging
import os

import numpy as np
import orjson

from src.common.config import VECTOR_STORE_CACHE_DIR

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    텍스트 내용의 해시를 키로 하여 임베딩 벡터를 캐싱합니다.
    """

    def __init__(self, cache_dir: str = VECTOR_STORE_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "embedding_cache.json")
        self._cache: dict[str, list[float]] = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    return orjson.loads(f.read())
            except Exception as e:
                logger.error(f"[Cache] 임베딩 캐시 로드 실패: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(orjson.dumps(self._cache))
        except Exception as e:
            logger.error(f"[Cache] 임베딩 캐시 저장 실패: {e}")

    def get_embeddings(
        self, texts: list[str]
    ) -> tuple[list[np.ndarray] | None, list[int]]:
        """
        텍스트 리스트에 대해 캐싱된 임베딩을 조회합니다.

        Returns:
            - 모든 텍스트의 임베딩이 캐시에 있을 경우 [벡터 리스트], 캐시 미스 발생 시 None
            - 캐시 미스가 발생한 텍스트의 인덱스 리스트
        """
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        results = [self._cache.get(h) for h in hashes]

        missing_indices = [i for i, res in enumerate(results) if res is None]

        if not missing_indices:
            return [np.array(r) for r in results], []

        return None, missing_indices

    def get_single_embedding(self, text: str) -> np.ndarray | None:
        """단일 텍스트의 캐시된 임베딩을 조회합니다."""
        h = hashlib.sha256(text.encode()).hexdigest()
        v = self._cache.get(h)
        return np.array(v) if v is not None else None

    def set_embeddings(self, texts: list[str], vectors: list[np.ndarray]) -> None:
        """
        텍스트와 생성된 임베딩 벡터를 캐시에 저장합니다.
        """
        for t, v in zip(texts, vectors, strict=False):
            h = hashlib.sha256(t.encode()).hexdigest()
            self._cache[h] = v.tolist()
        self._save_cache()
