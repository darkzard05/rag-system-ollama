"""
동적 모델 로더 - 비동기 모델 로드/언로드, 캐싱, 리소스 풀.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set
from threading import RLock
import time

logger = logging.getLogger(__name__)


class ModelLoadingStrategy(Enum):
    """모델 로딩 전략."""

    EAGER = "eager"  # 즉시 로드
    LAZY = "lazy"  # 필요 시 로드
    PRELOAD = "preload"  # 사전 예측 로드


class ModelEvictionPolicy(Enum):
    """모델 제거 정책."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


@dataclass
class ModelCacheConfig:
    """모델 캐시 설정."""

    max_models: int = 3
    ttl_seconds: float = 3600.0  # 1시간
    eviction_policy: ModelEvictionPolicy = ModelEvictionPolicy.LRU
    enable_persistence: bool = False


class ModelCache:
    """모델 캐시 - LRU/LFU 정책."""

    def __init__(self, config: ModelCacheConfig):
        self.config = config
        self._lock = RLock()
        self._cache: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}

    def get(self, model_key: str) -> Optional[Any]:
        """캐시에서 모델 조회."""
        with self._lock:
            if model_key not in self._cache:
                return None

            # TTL 확인
            load_time = self._load_times[model_key]
            if time.time() - load_time > self.config.ttl_seconds:
                logger.info(f"모델 TTL 만료: {model_key}")
                del self._cache[model_key]
                del self._load_times[model_key]
                del self._access_times[model_key]
                del self._access_counts[model_key]
                return None

            # 액세스 통계 업데이트
            self._access_times[model_key] = time.time()
            self._access_counts[model_key] = self._access_counts.get(model_key, 0) + 1

            return self._cache[model_key]

    def put(self, model_key: str, model: Any) -> bool:
        """캐시에 모델 저장.

        Returns:
            저장 성공 여부
        """
        with self._lock:
            # 용량 초과 시 제거
            if len(self._cache) >= self.config.max_models:
                if not self._evict():
                    logger.error(f"모델 제거 실패: {model_key}")
                    return False

            self._cache[model_key] = model
            self._load_times[model_key] = time.time()
            self._access_times[model_key] = time.time()
            self._access_counts[model_key] = 0

            return True

    def _evict(self) -> bool:
        """정책에 따라 모델 제거."""
        if not self._cache:
            return False

        if self.config.eviction_policy == ModelEvictionPolicy.LRU:
            # 가장 최근에 사용되지 않은 모델 제거
            lru_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )
        elif self.config.eviction_policy == ModelEvictionPolicy.LFU:
            # 가장 적게 사용된 모델 제거
            lfu_key = min(
                self._access_counts.keys(), key=lambda k: self._access_counts[k]
            )
            lru_key = lfu_key
        elif self.config.eviction_policy == ModelEvictionPolicy.FIFO:
            # 가장 먼저 로드된 모델 제거
            fifo_key = min(self._load_times.keys(), key=lambda k: self._load_times[k])
            lru_key = fifo_key
        else:
            lru_key = next(iter(self._cache.keys()))

        logger.info(f"모델 제거 (정책: {self.config.eviction_policy.value}): {lru_key}")

        del self._cache[lru_key]
        del self._load_times[lru_key]
        del self._access_times[lru_key]
        del self._access_counts[lru_key]

        return True

    def remove(self, model_key: str):
        """모델 명시적 제거."""
        with self._lock:
            if model_key in self._cache:
                del self._cache[model_key]
                del self._load_times[model_key]
                del self._access_times[model_key]
                del self._access_counts[model_key]

    def clear(self):
        """캐시 전체 정리."""
        with self._lock:
            self._cache.clear()
            self._load_times.clear()
            self._access_times.clear()
            self._access_counts.clear()

    def get_stats(self) -> Dict[str, int]:
        """캐시 통계."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.config.max_models,
                "utilization_percent": int(
                    100 * len(self._cache) / self.config.max_models
                ),
            }


class DynamicModelLoader:
    """동적 모델 로더."""

    def __init__(
        self,
        strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY,
        cache_config: Optional[ModelCacheConfig] = None,
    ):
        self.strategy = strategy
        self.cache_config = cache_config or ModelCacheConfig()
        self.cache = ModelCache(self.cache_config)

        self._lock = RLock()
        self._loading: Set[str] = set()  # 로딩 중인 모델
        self._loaded_models: Dict[str, Any] = {}  # 실제 로드된 모델
        self._load_count: Dict[str, int] = {}  # 로드 횟수

    async def load_model(
        self, model_key: str, timeout_seconds: float = 30.0
    ) -> Optional[Any]:
        """모델 로드 (캐시 우선).

        Args:
            model_key: 모델 키 (name:version)
            timeout_seconds: 로드 타임아웃

        Returns:
            로드된 모델 객체
        """
        # 캐시에서 확인
        cached_model = self.cache.get(model_key)
        if cached_model:
            logger.debug(f"캐시에서 모델 로드: {model_key}")
            with self._lock:
                self._load_count[model_key] = self._load_count.get(model_key, 0) + 1
            return cached_model

        # 이미 로딩 중이면 대기
        with self._lock:
            if model_key in self._loading:
                logger.debug(f"모델 로딩 중 (대기): {model_key}")

        # 로딩 시작
        with self._lock:
            self._loading.add(model_key)

        try:
            start_time = asyncio.get_event_loop().time()

            # 실제 모델 로드 시뮬레이션 (실제는 Ollama 호출)
            model = await self._load_model_async(model_key, timeout_seconds)

            if model:
                # 캐시에 저장
                if self.cache.put(model_key, model):
                    with self._lock:
                        self._loaded_models[model_key] = model
                        self._load_count[model_key] = (
                            self._load_count.get(model_key, 0) + 1
                        )

                    elapsed = asyncio.get_event_loop().time() - start_time
                    logger.info(f"모델 로드 완료: {model_key} ({elapsed:.2f}s)")
                    return model
        finally:
            with self._lock:
                self._loading.discard(model_key)

        return None

    async def _load_model_async(
        self, model_key: str, timeout_seconds: float
    ) -> Optional[Any]:
        """비동기 모델 로드 (구현 필요)."""
        # 이 부분은 실제 Ollama 통합 시 구현
        # 여기서는 시뮬레이션
        await asyncio.sleep(0.1)  # 로드 시뮬레이션
        return {"model_key": model_key, "loaded": True}

    async def unload_model(self, model_key: str):
        """모델 언로드."""
        with self._lock:
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
            self._load_count.pop(model_key, None)

        self.cache.remove(model_key)
        logger.info(f"모델 언로드: {model_key}")

    def get_loaded_models(self) -> Dict[str, Any]:
        """로드된 모든 모델 조회."""
        with self._lock:
            return dict(self._loaded_models)

    async def auto_unload_unused(self, keep_models: int = 1):
        """미사용 모델 자동 언로드."""
        with self._lock:
            loaded_keys = list(self._loaded_models.keys())

        # 로드 횟수가 적은 모델부터 언로드
        if len(loaded_keys) > keep_models:
            sorted_keys = sorted(loaded_keys, key=lambda k: self._load_count.get(k, 0))

            for model_key in sorted_keys[:-keep_models]:
                await self.unload_model(model_key)

    async def preload_models(self, model_keys: list):
        """여러 모델 사전 로드."""
        tasks = [self.load_model(key) for key in model_keys]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"사전 로드 완료: {len(model_keys)}개 모델")

    def get_cache_stats(self) -> Dict:
        """캐시 통계."""
        return self.cache.get_stats()

    async def clear_all(self):
        """모든 모델 언로드."""
        with self._lock:
            models_to_clear = list(self._loaded_models.keys())

        for model_key in models_to_clear:
            await self.unload_model(model_key)

        logger.info("모든 모델 언로드 완료")


# 전역 인스턴스
_loader_instance: Optional[DynamicModelLoader] = None


def get_dynamic_model_loader(
    strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY,
    cache_config: Optional[ModelCacheConfig] = None,
) -> DynamicModelLoader:
    """전역 동적 모델 로더 조회."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DynamicModelLoader(strategy, cache_config)
    return _loader_instance


def reset_dynamic_model_loader():
    """동적 모델 로더 리셋 (테스트용)."""
    global _loader_instance
    _loader_instance = None
