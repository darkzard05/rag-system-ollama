"""
ObjectCache + SyncCacheBridge - in-memory object cache and sync bridge.
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Coroutine
from threading import RLock
from typing import Any

from services.optimization._backends_common import CacheBackend, CacheEntry, R, T
from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)


class ObjectCache(CacheBackend[T]):
    """
    객체 캐시 - 임의의 메모리 파이썬 객체 보관 (컴파일된 그래프, 파싱된
    벡터스토어 등).

    MemoryCache와 동일한 LRU/제거/TTL/락 관용구를 따르되, 값을 unsafe 역직렬화
    하지 않고 메모리에 그대로 보관한다. CacheBackend ABC를 구현하므로 모든
    메서드는 비동기(async)이다.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 0.0,
    ):
        self.max_size = max_size
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

            if entry.is_expired():
                logger.debug(f"[ObjectCache] 만료된 항목 제거: {key}")
                del self.cache[key]
                self.stats.total_misses += 1
                self.stats.total_expirations += 1
                self.stats.cache_size = len(self.cache)
                return None

            entry.touch()
            self.stats.total_hits += 1
            self.stats.update_hit_rate()

            logger.debug(f"[ObjectCache] 캐시 히트: {key}")
            return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정 (객체는 직렬화 없이 보관)"""
        with self.lock:
            ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

            # 크기 초과 시 LRU 제거
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl,
            )

            self.cache[key] = entry
            self.stats.cache_size = len(self.cache)

            logger.debug(f"[ObjectCache] 값 저장: {key} (TTL: {ttl}초)")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.cache_size = len(self.cache)
                logger.debug(f"[ObjectCache] 값 삭제: {key}")

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.stats.cache_size = 0
            logger.info("[ObjectCache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회 (cache_size 정확도 보장)"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            ages = [entry.get_age() for entry in self.cache.values()]
            self.stats.avg_age_seconds = sum(ages) / len(ages) if ages else 0
            return self.stats

    def _evict_lru(self) -> None:
        """LRU 항목 제거"""
        if not self.cache:
            return

        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)
        del self.cache[lru_key]
        self.stats.total_evictions += 1
        self.stats.cache_size = len(self.cache)
        logger.debug(f"[ObjectCache] LRU 제거: {lru_key}")


class SyncCacheBridge:
    """
    동기식 브릿지 - 비동기 ObjectCache를 동기 컨텍스트(일반 def)에서 사용.

    Consumer인 VectorStoreCache.load/save는 평범한 def(비동기 아님)라서
    await할 수 없다. 따라서 브릿지만의 전용 이벤트 루프 + 데몬 스레드를
    두고, run_coroutine_threadsafe(...)로 비동기 메서드를 호출한다.

    전용 루프/스레드 방식은 호출자가 동기 컨텍스트에 있든 이미 실행 중인
    다른 이벤트 루프 안에 있든 어디서든 안전하다. (loop.run_until_complete는
    호출자가 이미 실행 중인 루프 안에 있으면 "This event loop is already
    running" RuntimeError를 발생시키므로 사용하지 않는다.)
    """

    def __init__(self, cache: ObjectCache[T]):
        self._cache: ObjectCache[Any] = cache
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="SyncCacheBridge"
        )
        self._thread.start()

    def _run(self, coro: Coroutine[Any, Any, R]) -> R:
        """전용 루프/스레드에서 코루틴 실행 (호출자 루프 컨텍스트 무관)."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close_sync(self) -> None:
        """브릿지 전용 루프/스레드 정리."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            # 스레드가 run_forever를 빠져나올 때까지 대기 후 close().
            self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()

    def __del__(self) -> None:
        # __del__ 내부 리소스 정리 보조: 예외는 삼킨다.
        with contextlib.suppress(Exception):
            self.close_sync()

    def get_sync(self, key: str) -> T | None:
        """동기 조회"""
        result: T | None = self._run(self._cache.get(key))
        return result

    def set_sync(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """동기 설정"""
        self._run(self._cache.set(key, value, ttl_seconds))

    def delete_sync(self, key: str) -> None:
        """동기 삭제"""
        self._run(self._cache.delete(key))

    def clear_sync(self) -> None:
        """동기 전체 삭제"""
        self._run(self._cache.clear())
