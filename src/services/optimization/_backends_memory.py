"""
MemoryCache backend - LRU/TTL in-memory cache.
"""

import logging
import time
from threading import RLock

from services.optimization._backends_common import CacheBackend, CacheEntry, T
from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)


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
        """값 설정 (사이즈 추적 최적화)"""
        with self.lock:
            ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

            # [최적화] 기존 항목이 있으면 사이즈 차감
            if key in self.cache:
                self.stats.total_memory_bytes -= self.cache[key].metadata.get(
                    "size_bytes", 0
                )

            # 메모리 확인 및 정리
            self._cleanup_if_needed(value)

            # 대략적인 사이즈 계산
            import sys

            size_bytes = sys.getsizeof(value)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl,
                metadata={"size_bytes": size_bytes},
            )

            self.cache[key] = entry
            self.stats.total_memory_bytes += size_bytes
            self.stats.cache_size = len(self.cache)

            logger.debug(f"[Cache] 값 저장: {key} (TTL: {ttl}초, Size: {size_bytes}B)")

    async def delete(self, key: str) -> None:
        """값 삭제 (사이즈 차감 포함)"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_memory_bytes -= entry.metadata.get("size_bytes", 0)
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
        """통계 조회 (계산 오버헤드 최적화)"""
        with self.lock:
            stats = self.stats
            stats.cache_size = len(self.cache)

            # [최적화] 모든 항목을 순회하며 JSON 직렬화를 반복하는 대신,
            # 저장 시 계산된 total_size_bytes를 즉시 활용
            ages = [entry.get_age() for entry in self.cache.values()]
            stats.avg_age_seconds = sum(ages) / len(ages) if ages else 0

            return stats

    def _cleanup_if_needed(self, new_value: T) -> None:
        """메모리 및 크기 조건에 따라 정리 (계산 최적화)"""
        # 1. 크기 초과 확인 (O(1))
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # 2. 메모리 초과 확인
        # [최적화] 매번 전체 캐시를 순회하지 않고, 새로 추가될 값의 크기만 계산
        try:
            # sys.getsizeof()는 실제 메모리 점유율을 정확히 반영하지 못하므로
            # 직렬화된 크기를 기준으로 하되, 이미 계산된 total_size_bytes를 활용
            import sys

            estimated_bytes = sys.getsizeof(new_value)
            if isinstance(new_value, (str, bytes)):
                estimated_bytes = len(new_value)
            elif isinstance(new_value, (list, dict, set, tuple)):
                estimated_bytes = len(new_value) * 8  # 대략적인 포인터 크기
            else:
                estimated_bytes = 1024
        except Exception:
            estimated_bytes = 1024  # 폴백: 1KB

        if (
            self.stats.total_memory_bytes + estimated_bytes
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
