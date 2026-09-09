"""
캐싱 최적화 - Task 13
응답 캐싱, 세맨틱 캐싱, TTL 관리, 캐시 일관성
"""

import logging
from threading import RLock
from typing import Any

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization._backends import (  # noqa: F401 — re-exports for backward compat
    CACHE_FORMAT,
    CACHE_FORMAT_KEY,
    CacheBackend,
    CacheEntry,
    DiskCache,
    MemoryCache,
    ObjectCache,
    R,
    SemanticCache,
    SyncCacheBridge,
    T,
    _json_default,
)
from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)


# ============================================================================
# Module layout (caching-optimizer split)
# ----------------------------------------------------------------------------
# 캐시 백엔드 계층(CacheEntry/CacheBackend/MemoryCache/SemanticCache/DiskCache/
# ObjectCache/SyncCacheBridge + CACHE_FORMAT/CACHE_FORMAT_KEY/_json_default)은
# ``services.optimization._backends`` 로, 통계(CacheStatistics)는
# ``services.optimization._metrics`` 로 분리되었다. 위 import 는 분리 이전
# import 경로(``services.optimization.caching_optimizer.<symbol>``)에 대한 하위
# 호환 재수출(re-export)이다. 이 모듈에는 CacheManager 와 get_cache_manager 만
# 정의되어 있다.
# ============================================================================


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
        enable_disk_cache: bool = True,
        embedding_model=None,
        memory_cache_size: int = 1000,
        semantic_cache_size: int = 500,
        disk_cache_dir: str = "./.model_cache/response_cache",
    ):
        self.enable_memory_cache = enable_memory_cache
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_disk_cache = enable_disk_cache

        self.memory_cache: MemoryCache | None = None
        self.semantic_cache: SemanticCache | None = None
        self.disk_cache: DiskCache | None = None

        if enable_memory_cache:
            self.memory_cache = MemoryCache(max_size=memory_cache_size)

        if enable_semantic_cache:
            self.semantic_cache = SemanticCache(
                embedding_model=embedding_model, max_entries=semantic_cache_size
            )

        if enable_disk_cache:
            self.disk_cache = DiskCache(cache_dir=disk_cache_dir)

        self.lock = RLock()

    async def get(self, key: str, use_semantic: bool = False) -> Any | None:
        """값 조회 (L1 -> L2 -> L3)"""
        with get_performance_monitor().track_operation(
            OperationType.QUERY_PROCESSING,
            {"stage": "cache_lookup", "semantic": use_semantic},
        ) as op:
            # 1. L1 메모리 캐시 확인 (가장 빠름)
            if self.memory_cache and not use_semantic:
                result = await self.memory_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L1"}
                    return result

            # 2. L2 세맨틱 캐시 확인 (의미적 유사성)
            if self.semantic_cache and use_semantic:
                result = await self.semantic_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L2"}

                    # L1으로 승격 (Promotion)
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            # 3. L3 디스크 캐시 확인 (영구 저장소)
            if self.disk_cache and not use_semantic:
                result = await self.disk_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L3"}

                    # L1으로 승격
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float = 0,
        use_semantic: bool = False,
        persist_to_disk: bool = True,
    ) -> None:
        """값 저장 (L1, L2, L3)"""
        with self.lock:
            # L1에 저장
            if self.memory_cache:
                await self.memory_cache.set(key, value, ttl_seconds)

            # L2 (세맨틱) 저장
            if self.semantic_cache and use_semantic:
                await self.semantic_cache.set(key, value, ttl_seconds)

            # L3 (디스크) 저장
            # [최적화] persist_to_disk=False 시 디스크 저장 생략
            # (문장 임베딩은 FAISS 벡터 캐시에 영속화되므로 중복 디스크 저장 불필요)
            if self.disk_cache and persist_to_disk:
                await self.disk_cache.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        """전체 레이어에서 삭제"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.delete(key)
            if self.semantic_cache:
                await self.semantic_cache.delete(key)
            if self.disk_cache:
                await self.disk_cache.delete(key)

    async def clear(self) -> None:
        """모든 캐시 비우기"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.clear()
            if self.semantic_cache:
                await self.semantic_cache.clear()
            if self.disk_cache:
                await self.disk_cache.clear()

    def get_stats(self) -> dict[str, CacheStatistics]:
        """레이어별 통계"""
        stats = {}
        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats()
        if self.semantic_cache:
            stats["semantic"] = self.semantic_cache.get_stats()
        if self.disk_cache:
            stats["disk"] = self.disk_cache.get_stats()
        return stats

    def get_combined_stats(self) -> CacheStatistics:
        """통합 통계"""
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
    enable_disk_cache: bool = True,
    embedding_model=None,
) -> CacheManager:
    """캐시 관리자 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            enable_memory_cache=enable_memory_cache,
            enable_semantic_cache=enable_semantic_cache,
            enable_disk_cache=enable_disk_cache,
            embedding_model=embedding_model,
        )
    return _cache_manager
