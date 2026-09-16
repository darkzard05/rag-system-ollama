"""SemanticCache backend - embedding-based similarity cache."""

import hashlib
import logging
import time
from threading import RLock

import numpy as np

from common.similarity import normalize_vector
from common.utils import fast_hash
from services.optimization._backends_common import CacheBackend, CacheEntry, T
from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)


class SemanticCache(CacheBackend[T]):
    """Semantic cache - embedding-based similarity search."""

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.95,
        max_entries: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.default_ttl = ttl_seconds
        self.embeddings: dict[str, np.ndarray] = {}
        self.cache: dict[str, CacheEntry] = {}
        self.lock = RLock()
        self.stats = CacheStatistics()
        self._cached_matrix: np.ndarray | None = None
        self._cached_keys: list[str] = []

    @property
    def cache_size(self) -> int:
        """캐시 크기"""
        return len(self.cache)

    def _update_matrix(self, key: str | None = None, action: str = "add") -> None:
        """캐시된 행렬 업데이트 (증분 방식)"""
        if action == "add" and key is not None:
            self._cached_keys.append(key)
            new_embedding = self.embeddings[key]
            if self._cached_matrix is None:
                self._cached_matrix = np.array([new_embedding])
            else:
                self._cached_matrix = np.vstack([self._cached_matrix, new_embedding])
        elif action == "remove" and key is not None:
            try:
                idx = self._cached_keys.index(key)
                self._cached_keys.pop(idx)
                if self._cached_matrix is not None:
                    self._cached_matrix = np.delete(self._cached_matrix, idx, axis=0)
                    if self._cached_matrix.size == 0:
                        self._cached_matrix = None
            except ValueError:
                pass
        else:
            self._cached_keys = list(self.embeddings.keys())
            if not self._cached_keys:
                self._cached_matrix = None
            else:
                self._cached_matrix = np.array(
                    [self.embeddings[k] for k in self._cached_keys]
                )

    async def _embed(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        if self.embedding_model is not None and hasattr(
            self.embedding_model, "embed_query"
        ):
            embedding = await self.embedding_model.embed_query(text)
            return np.array(embedding)

        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int % (2**32))
        return np.random.randn(384)

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            self.stats.update_hit_rate()
            return self.stats

    async def get(
        self,
        key: str,
        similarity_threshold: float | None = None,
        query_embedding: list | None = None,
    ) -> T | None:
        """
        의미적으로 유사한 항목 조회 (NumPy 벡터화 최적화)

        B10 최적화: query_embedding 제공 시 재임베딩을 생략한다.
        미제공 시에는 기존 계약대로 내부 ``_embed`` 로 임베딩한다 (B10 회귀 수정).
        """
        if query_embedding is None:
            # B10 회귀 수정: 임베딩을 제공하지 않은 호출(기존 API)은
            # 내부 재임베딩으로 정상 동작해야 한다 — 캐시 비활성화가 아니다.
            if not self.embedding_model or not self.embeddings:
                return None
            embed_vec = await self._embed(key)
            query_vec = normalize_vector(embed_vec, eps=1e-10)
        else:
            # 호출자 제공 벡터 사용 — `_embed` 호출 없음
            query_vec = normalize_vector(
                np.asarray(query_embedding, dtype=np.float64), eps=1e-10
            )

        with self.lock:
            threshold = similarity_threshold or self.similarity_threshold

            try:
                # [최적화] 캐시된 행렬 사용
                if self._cached_matrix is None:
                    self._update_matrix()

                if self._cached_matrix is None:
                    return None

                # 코사인 유사도 계산 (행렬-벡터 내적)
                similarities = np.dot(self._cached_matrix, query_vec)

                # 가장 유사한 항목 찾기
                max_idx = np.argmax(similarities)
                best_similarity = similarities[max_idx]
                best_match = self._cached_keys[max_idx]

                # 임계값 이상인 경우 반환
                if best_similarity >= threshold:
                    entry = self.cache[best_match]
                    if entry.is_expired():
                        self._update_matrix(best_match, action="remove")
                        del self.cache[best_match]
                        self.embeddings.pop(best_match, None)
                        self.stats.total_misses += 1
                        self.stats.total_expirations += 1
                        logger.debug(f"[SemanticCache] 만료된 항목 제거: {best_match}")
                        return None
                    entry.touch()
                    self.stats.total_hits += 1
                    logger.debug(f"[SemanticCache] 히트: 유사도 {best_similarity:.3f}")
                    return entry.value

                self.stats.total_misses += 1
                return None

            except Exception as e:
                logger.error(f"[SemanticCache] 조회 오류: {e}")
                self.stats.total_misses += 1
                return None

    async def set(
        self,
        key: str,
        value: T,
        ttl_seconds: float = 0,
        query_embedding: list | None = None,
    ) -> None:
        """값 저장 및 벡터 정규화

        B10 최적화: query_embedding 제공 시 재임베딩을 생략한다.
                미제공 시에는 기존 계약대로 내부 ``_embed`` 로 임베딩한다 (B10 회귀 수정).
        """
        if query_embedding is None:
            # 미설정 embedder여도 _embed는 해시 기반 벡터를 항상 반환하므로
            # 재임베딩 없이는 저장하지 않는 B10 계약은 유지하지 않는다.
            query_vec = np.asarray(await self._embed(key), dtype=np.float64)
        else:
            query_vec = np.asarray(query_embedding, dtype=np.float64)

        with self.lock:
            try:
                ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

                if len(self.cache) >= self.max_entries:
                    self._evict_oldest()

                cache_key = fast_hash(key)

                # [최적화] 저장 시 미리 정규화하여 get 단계의 연산 감소
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm

                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl,
                    metadata={"query": key[:100]},
                )

                self.cache[cache_key] = entry
                self.embeddings[cache_key] = query_vec
                self._update_matrix()  # 행렬 업데이트
                self.stats.cache_size = len(self.cache)

            except Exception as e:
                logger.error(f"[SemanticCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.embeddings:
                del self.embeddings[key]
                self._update_matrix()  # 행렬 업데이트
            self.stats.cache_size = len(self.cache)

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.embeddings.clear()
            self._cached_matrix = None
            self._cached_keys = []
            self.stats.cache_size = 0
            logger.info("[SemanticCache] 캐시 전체 삭제")

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거"""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)

        del self.cache[oldest_key]
        if oldest_key in self.embeddings:
            del self.embeddings[oldest_key]
            self._update_matrix()  # 행렬 업데이트

        self.stats.total_evictions += 1
        logger.debug(f"[SemanticCache] 가장 오래된 항목 제거: {oldest_key}")
