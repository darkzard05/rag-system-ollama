"""
메모리 최적화 통합 모듈 - 프로파일링, GC, 리소스 관리 통합.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from threading import RLock, Thread
import time

from services.monitoring.memory_profiler import (
    MemoryProfiler,
    MemoryMonitor,
    get_memory_profiler,
    get_memory_monitor,
)
from infra.resource_manager import (
    ResourceManager,
    ContextualResourceManager,
    get_resource_manager,
    get_contextual_resource_manager,
)

logger = logging.getLogger(__name__)


class MemoryOptimizationMode(Enum):
    """메모리 최적화 모드."""
    NORMAL = "normal"
    STRICT = "strict"        # 메모리 절감 우선
    PERFORMANCE = "performance"  # 성능 우선


@dataclass
class MemoryOptimizationConfig:
    """메모리 최적화 설정."""
    mode: MemoryOptimizationMode = MemoryOptimizationMode.NORMAL
    enable_monitoring: bool = False # 중복 로그 방지를 위해 기본 모니터링은 끔 (Optimizer 루프에서 수행)
    enable_adaptive_gc: bool = True
    enable_resource_pooling: bool = True
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0


class MemoryOptimizer:
    """메모리 최적화 통합 관리자."""
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        self.config = config or MemoryOptimizationConfig()
        
        self._lock = RLock()
        self._is_running = False
        self._monitor_thread: Optional[Thread] = None
        
        # 컴포넌트 초기화
        self.profiler = get_memory_profiler()
        self.monitor = get_memory_monitor() if self.config.enable_monitoring else None
        self.resource_manager = (
            get_resource_manager()
            if self.config.enable_resource_pooling
            else None
        )
        self.contextual_resource_manager = (
            get_contextual_resource_manager()
            if self.config.enable_resource_pooling
            else None
        )
        
        self._stats_history: List[Dict] = []
    
    def start(self):
        """메모리 최적화 시작."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # [최적화] 중복 모니터링 스레드 방지
        # if self.monitor:
        #     self.monitor.start_monitoring()
        
        self._monitor_thread = Thread(
            target=self._optimize_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("[Monitor] [Memory] 메모리 최적화 서비스 시작됨")
    
    def stop(self):
        """메모리 최적화 중지."""
        self._is_running = False
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("[Monitor] [Memory] 메모리 최적화 서비스 중지됨")
    
    def _optimize_loop(self):
        """최적화 루프."""
        from core.thread_safe_session import ThreadSafeSessionManager
        while self._is_running:
            try:
                # [안전 장치] 답변 생성 중에는 최적화 로직 건너뜀 (전역 플래그 사용)
                if ThreadSafeSessionManager._is_generating_globally:
                    time.sleep(2)
                    continue

                metrics = self.profiler.get_current_memory()
                
                # 현재 상태 저장
                with self._lock:
                    self._stats_history.append({
                        "timestamp": time.time(),
                        "memory_mb": metrics.current_mb,
                        "percent_used": metrics.percent_used,
                        "num_objects": metrics.num_objects,
                        "growth_rate": metrics.growth_rate_mb_per_sec,
                    })
                    
                    # 최근 1000개만 유지
                    if len(self._stats_history) > 1000:
                        self._stats_history.pop(0)
                
                # 누수 감지
                leaks = self.profiler.detect_memory_leaks()
                if leaks:
                    for leak in leaks:
                        logger.debug(
                            f"메모리 사용량 증가 감지 (정상 범위 확인 중): {leak.object_type}, "
                            f"증가율 {leak.growth_rate:.2f} MB/min"
                        )
                
                time.sleep(5)  # 5초마다 체크
            except Exception as e:
                logger.error(f"최적화 루프 에러: {e}")
    
    def get_memory_stats(self) -> Dict:
        """메모리 통계 조회."""
        metrics = self.profiler.get_current_memory()
        
        return {
            "memory": {
                "current_mb": metrics.current_mb,
                "peak_mb": metrics.peak_mb,
                "available_mb": metrics.available_mb,
                "percent_used": metrics.percent_used,
                "growth_rate_mb_per_sec": metrics.growth_rate_mb_per_sec,
            },
            "objects": {
                "num_objects": metrics.num_objects,
                "num_tracked": metrics.num_tracked_objects,
            },
            "mode": self.config.mode.value,
        }
    
    def get_memory_history(self, last_n: int = 100) -> List[Dict]:
        """메모리 히스토리 조회."""
        with self._lock:
            return self._stats_history[-last_n:]
    
    def get_top_memory_users(self, top_n: int = 10) -> List[tuple]:
        """메모리 사용량이 큰 상위 객체 조회."""
        return self.profiler.get_top_objects(top_n)
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List:
        """메모리 누수 감지."""
        return self.profiler.detect_memory_leaks(threshold_mb)
    
    async def create_resource_pool(
        self,
        name: str,
        resource_type,
        max_size: int = 10,
    ):
        """리소스 풀 생성."""
        if not self.resource_manager:
            logger.warning("리소스 풀링이 비활성화됨")
            return None
        
        return self.resource_manager.create_pool(name, resource_type, max_size)
    
    def get_resource_stats(self) -> Dict:
        """리소스 통계 조회."""
        if not self.resource_manager:
            return {}
        
        return {
            name: {
                "total_created": stats.total_created,
                "currently_allocated": stats.currently_allocated,
                "total_released": stats.total_released,
                "peak_usage": stats.peak_usage,
            }
            for name, stats in self.resource_manager.get_stats().items()
        }


class MemoryOptimizationContext:
    """메모리 최적화 컨텍스트 - with 문 지원."""
    
    def __init__(self, optimizer: MemoryOptimizer):
        self.optimizer = optimizer
    
    def __enter__(self):
        self.optimizer.start()
        return self.optimizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.optimizer.stop()


# 전역 인스턴스
_optimizer_instance: Optional[MemoryOptimizer] = None


def get_memory_optimizer(config: Optional[MemoryOptimizationConfig] = None) -> MemoryOptimizer:
    """전역 메모리 최적화 관리자 조회."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = MemoryOptimizer(config)
    return _optimizer_instance


def reset_memory_optimizer():
    """메모리 최적화 관리자 리셋 (테스트용)."""
    global _optimizer_instance
    if _optimizer_instance:
        _optimizer_instance.stop()
    _optimizer_instance = None
