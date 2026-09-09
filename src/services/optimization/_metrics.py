"""
캐시 통계 계층 - Task 13 캐싱 최적화

caching_optimizer 모듈에서 분리된 캐시 통계(CacheStatistics)가 위치한다.
기존 import 경로(``services.optimization.caching_optimizer.CacheStatistics``)는
caching_optimizer 모듈의 재수출로 유지된다.
"""

from dataclasses import dataclass


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
