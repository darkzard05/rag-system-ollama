"""
캐싱 최적화 - Task 13
응답 캐싱, 세맨틱 캐싱, TTL 관리, 캐시 일관성
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Generic, TypeVar

import numpy as np

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

# 타입 변수
T = TypeVar("T")


@dataclass
class CacheEntry:
    """캐시 항목"""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: float
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """TTL 만료 여부 확인"""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def get_age(self) -> float:
        """항목 나이 (초)"""
        return time.time() - self.created_at

    def touch(self) -> None:
        """접근 시간 업데이트"""
        self.accessed_at = time.time()
        self.hit_count += 1


@dataclass
class CacheStatistics:
    """캐시 통계"""

    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    cache_size: int = 0
    total_memory_bytes: int = 0
    hit_rate: float = 0.0
    avg_age_seconds: float = 0.0

    @property
    def total_requests(self) -> int:
        return self.total_hits + self.total_misses

    def update_hit_rate(self) -> None:
        """히트율 계산"""
        if self.total_requests > 0:
            self.hit_rate = self.total_hits / self.total_requests


class CacheBackend(ABC, Generic[T]):
    """캐시 백엔드 추상 클래스"""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """값 조회"""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """값 삭제"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """캐시 전체 삭제"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        pass


class MemoryCache(CacheBackend[T]):
    """
    메모리 기반 캐시

    특징:
    - LRU 제거 정책
    - TTL 만료 처리
    - 메모리 사용량 추적
    - 통계 수집
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.default_ttl = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}
        self.lock = RLock()
        self.stats = CacheStatistics()

    async def get(self, key: str) -> T | None:
        """값 조회"""
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.stats.total_misses += 1
                return None

            # 만료 확인
            if entry.is_expired():
                logger.debug(f"[Cache] 만료된 항목 제거: {key}")
                del self.cache[key]
                self.stats.total_misses += 1
                self.stats.total_expirations += 1
                return None

            # 접근 업데이트
            entry.touch()
            self.stats.total_hits += 1
            self.stats.update_hit_rate()

            logger.debug(f"[Cache] 캐시 히트: {key} (히트 수: {entry.hit_count})")
            return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정"""
        with self.lock:
            ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

            # 메모리 확인 및 정리
            self._cleanup_if_needed(value)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl,
            )

            self.cache[key] = entry
            self.stats.cache_size = len(self.cache)

            logger.debug(f"[Cache] 값 저장: {key} (TTL: {ttl}초)")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.cache_size = len(self.cache)
                logger.debug(f"[Cache] 값 삭제: {key}")

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.stats.cache_size = 0
            logger.info("[Cache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            stats = self.stats
            stats.cache_size = len(self.cache)

            # 메모리 사용량 계산
            total_bytes = 0
            ages = []

            for entry in self.cache.values():
                total_bytes += len(json.dumps(entry.value).encode())
                ages.append(entry.get_age())

            stats.total_memory_bytes = total_bytes
            stats.avg_age_seconds = sum(ages) / len(ages) if ages else 0

            return stats

    def _cleanup_if_needed(self, new_value: T) -> None:
        """메모리 및 크기 조건에 따라 정리"""
        # 크기 초과 확인
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # 메모리 초과 확인
        try:
            estimated_bytes = len(json.dumps(new_value).encode())
        except (TypeError, ValueError):
            # JSON 직렬화 불가능한 객체는 크기 추정
            estimated_bytes = len(str(new_value).encode())

        stats = self.get_stats()

        if (
            stats.total_memory_bytes + estimated_bytes
            > self.max_memory_mb * 1024 * 1024
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        """LRU 항목 제거"""
        if not self.cache:
            return

        # 접근 시간이 가장 오래된 항목 찾기
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)

        del self.cache[lru_key]
        self.stats.total_evictions += 1
        logger.debug(f"[Cache] LRU 제거: {lru_key}")


class SemanticCache(CacheBackend[T]):
    """
    세맨틱 캐시 - 임베딩 기반 유사성 캐싱

    특징:
    - 벡터 기반 유사성 검색
    - 의미적으로 유사한 쿼리 매칭
    - 거리 기반 검색
    - 메모리 효율적
    """

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

    @property
    def cache_size(self) -> int:
        """캐시 크기"""
        return len(self.cache)

    async def get(
        self, query: str, similarity_threshold: float | None = None
    ) -> T | None:
        """
        의미적으로 유사한 항목 조회

        Args:
            query: 검색 쿼리
            similarity_threshold: 유사도 임계값 (None이면 기본값 사용)

        Returns:
            캐시된 값 또는 None
        """
        if not self.embedding_model:
            return None

        with self.lock:
            threshold = similarity_threshold or self.similarity_threshold

            try:
                # 쿼리 임베딩
                query_embedding = await self._embed(query)

                # 가장 유사한 항목 찾기
                best_match = None
                best_similarity: float = 0.0

                for key, cached_embedding in self.embeddings.items():
                    entry = self.cache.get(key)
                    if entry is None or entry.is_expired():
                        continue

                    # 코사인 유사도 계산
                    similarity = self._cosine_similarity(
                        query_embedding, cached_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key

                # 임계값 이상인 경우 반환
                if best_similarity >= threshold and best_match:
                    entry = self.cache[best_match]
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

    async def set(self, query: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정"""
        with self.lock:
            try:
                ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

                # 메모리 정리
                if len(self.cache) >= self.max_entries:
                    self._evict_oldest()

                # 캐시 키 생성 (해시)
                cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]

                # 쿼리 임베딩 (모델이 있으면)
                query_embedding = None
                if self.embedding_model:
                    try:
                        query_embedding = await self._embed(query)
                    except Exception as e:
                        logger.warning(f"[SemanticCache] 임베딩 생성 오류: {e}")

                # 항목 저장
                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl,
                    metadata={"query": query[:100]},
                )

                self.cache[cache_key] = entry
                if query_embedding is not None:
                    self.embeddings[cache_key] = query_embedding
                self.stats.cache_size = len(self.cache)

                logger.debug(f"[SemanticCache] 값 저장: {cache_key}")

            except Exception as e:
                logger.error(f"[SemanticCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.embeddings:
                del self.embeddings[key]
            self.stats.cache_size = len(self.cache)

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.embeddings.clear()
            self.stats.cache_size = 0
            logger.info("[SemanticCache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            self.stats.update_hit_rate()
            return self.stats

    async def _embed(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        # 실제 구현에서는 임베딩 모델 사용
        # 현재는 간단한 시뮬레이션
        if hasattr(self.embedding_model, "embed_query"):
            embedding = await self.embedding_model.embed_query(text)
            return np.array(embedding)

        # 폴백: 간단한 해시 기반 벡터
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int % (2**32))
        return np.random.randn(384)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거"""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)

        del self.cache[oldest_key]
        if oldest_key in self.embeddings:
            del self.embeddings[oldest_key]

        self.stats.total_evictions += 1
        logger.debug(f"[SemanticCache] 가장 오래된 항목 제거: {oldest_key}")


class CacheManager:
    """
    캐시 관리자 - 다중 캐시 백엔드 통합

    특징:
    - 다중 캐시 레이어 (L1: 메모리, L2: 세맨틱)
    - 자동 캐시 선택
    - 캐시 동기화
    - 통합 통계
    """

    def __init__(
        self,
        enable_memory_cache: bool = True,
        enable_semantic_cache: bool = True,
        embedding_model=None,
        memory_cache_size: int = 1000,
        semantic_cache_size: int = 500,
    ):
        self.enable_memory_cache = enable_memory_cache
        self.enable_semantic_cache = enable_semantic_cache

        self.memory_cache: MemoryCache | None = None
        self.semantic_cache: SemanticCache | None = None

        if enable_memory_cache:
            self.memory_cache = MemoryCache(max_size=memory_cache_size)

        if enable_semantic_cache:
            self.semantic_cache = SemanticCache(
                embedding_model=embedding_model, max_entries=semantic_cache_size
            )

        self.lock = RLock()

    async def get(self, key: str, use_semantic: bool = False) -> Any | None:
        """값 조회"""
        with monitor.track_operation(
            OperationType.QUERY_PROCESSING,
            {"stage": "cache_lookup", "semantic": use_semantic},
        ) as op:
            # L1 메모리 캐시 확인
            if self.memory_cache and not use_semantic:
                result = await self.memory_cache.get(key)
                if result is not None:
                    logger.debug(f"[CacheManager] L1 히트: {key}")
                    op.metadata = {"cache_level": "L1"}
                    return result

            # L2 세맨틱 캐시 확인
            if self.semantic_cache and use_semantic:
                result = await self.semantic_cache.get(key)
                if result is not None:
                    logger.debug(f"[CacheManager] L2 히트: {key}")
                    op.metadata = {"cache_level": "L2"}

                    # L1에도 저장
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            logger.debug(f"[CacheManager] 캐시 미스: {key}")
            return None

    async def set(
        self, key: str, value: Any, ttl_seconds: float = 0, use_semantic: bool = False
    ) -> None:
        """값 설정"""
        with self.lock:
            # L1에 저장
            if self.memory_cache:
                await self.memory_cache.set(key, value, ttl_seconds)

            # L2에도 저장
            if self.semantic_cache and use_semantic:
                await self.semantic_cache.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.delete(key)
            if self.semantic_cache:
                await self.semantic_cache.delete(key)

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.clear()
            if self.semantic_cache:
                await self.semantic_cache.clear()

    def get_stats(self) -> dict[str, CacheStatistics]:
        """통합 통계"""
        stats = {}

        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats()

        if self.semantic_cache:
            stats["semantic"] = self.semantic_cache.get_stats()

        return stats

    def get_combined_stats(self) -> CacheStatistics:
        """통합 통계 (전체)"""
        combined = CacheStatistics()

        for cache_stats in self.get_stats().values():
            combined.total_hits += cache_stats.total_hits
            combined.total_misses += cache_stats.total_misses
            combined.total_evictions += cache_stats.total_evictions
            combined.total_expirations += cache_stats.total_expirations
            combined.cache_size += cache_stats.cache_size
            combined.total_memory_bytes += cache_stats.total_memory_bytes

        combined.update_hit_rate()
        return combined


# 전역 캐시 관리자 인스턴스
_cache_manager: CacheManager | None = None


def get_cache_manager(
    enable_memory_cache: bool = True,
    enable_semantic_cache: bool = True,
    embedding_model=None,
) -> CacheManager:
    """캐시 관리자 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            enable_memory_cache=enable_memory_cache,
            enable_semantic_cache=enable_semantic_cache,
            embedding_model=embedding_model,
        )
    return _cache_manager


def reset_cache_manager() -> None:
    """캐시 관리자 리셋"""
    global _cache_manager
    _cache_manager = None
