"""
GC (Garbage Collection) 튜닝 및 최적화 모듈.
"""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import RLock

logger = logging.getLogger(__name__)


class GCStrategy(Enum):
    """GC 전략."""
    AGGRESSIVE = "aggressive"  # 자주 실행, 낮은 지연
    BALANCED = "balanced"      # 균형
    LAZY = "lazy"              # 적게 실행, 높은 지연


@dataclass
class GCConfig:
    """GC 설정."""
    strategy: GCStrategy = GCStrategy.BALANCED
    threshold0: int = 700      # 0세대 임계값
    threshold1: int = 10       # 1세대 임계값
    threshold2: int = 10       # 2세대 임계값
    collect_interval: float = 1.0  # 수동 collect 간격 (초)
    enable_debug: bool = False


@dataclass
class GCStats:
    """GC 통계."""
    collections: List[int]
    collected: List[int]
    uncollectable: List[int]
    collection_time_ms: float
    next_collection_ms: float


class GCTuner:
    """GC 튜닝 엔진."""
    
    def __init__(self, config: GCConfig):
        self.config = config
        self._lock = RLock()
        self._last_collect = 0.0
        self._collection_times: List[float] = []
        
        self._apply_strategy()
    
    def _apply_strategy(self):
        """GC 전략 적용."""
        if self.config.strategy == GCStrategy.AGGRESSIVE:
            # 자주 수집, 낮은 임계값
            gc.set_threshold(400, 5, 5)
        elif self.config.strategy == GCStrategy.BALANCED:
            # 기본값
            gc.set_threshold(
                self.config.threshold0,
                self.config.threshold1,
                self.config.threshold2
            )
        elif self.config.strategy == GCStrategy.LAZY:
            # 적게 수집, 높은 임계값
            gc.set_threshold(1000, 20, 20)
        
        if self.config.enable_debug:
            gc.set_debug(gc.DEBUG_STATS)
            logger.info("GC 디버그 활성화")
    
    def get_thresholds(self) -> Tuple[int, int, int]:
        """현재 GC 임계값 조회."""
        return gc.get_threshold()
    
    def set_thresholds(self, gen0: int, gen1: int, gen2: int):
        """GC 임계값 설정."""
        gc.set_threshold(gen0, gen1, gen2)
        logger.info(f"GC 임계값 설정: {gen0}, {gen1}, {gen2}")
    
    def force_collect(self, generations: Optional[int] = None) -> int:
        """강제 GC 실행.
        
        Args:
            generations: 수집할 세대 (None=모든 세대)
        
        Returns:
            수집된 객체 수
        """
        import time
        
        start_time = time.time()
        
        if generations is None:
            collected = gc.collect()
        else:
            collected = gc.collect(generations)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self._collection_times.append(elapsed_ms)
            if len(self._collection_times) > 100:
                self._collection_times.pop(0)
        
        logger.info(f"GC 수집: {collected}개 객체, {elapsed_ms:.1f}ms")
        
        return collected
    
    def get_stats(self) -> GCStats:
        """GC 통계 조회."""
        stats = gc.get_stats()
        
        collections = []
        collected = []
        uncollectable = []
        
        for stat in stats:
            collections.append(stat.get("collections", 0))
            collected.append(stat.get("collected", 0))
            uncollectable.append(stat.get("uncollectable", 0))
        
        avg_collection_time = (
            sum(self._collection_times) / len(self._collection_times)
            if self._collection_times
            else 0.0
        )
        
        return GCStats(
            collections=collections,
            collected=collected,
            uncollectable=uncollectable,
            collection_time_ms=avg_collection_time,
            next_collection_ms=gc.get_count()[0] / 10.0,  # 추정값
        )
    
    def disable_gc(self):
        """GC 비활성화 (성능 향상, 메모리 증가)."""
        gc.disable()
        logger.warning("GC 비활성화됨")
    
    def enable_gc(self):
        """GC 활성화."""
        gc.enable()
        logger.info("GC 활성화됨")
    
    def get_gc_objects(self) -> Dict[str, int]:
        """GC 추적 객체 분류."""
        objects = gc.get_objects()
        type_counts = {}
        
        for obj in objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        return dict(sorted(
            type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])  # 상위 20개
    
    def find_referrers(self, obj_type: str, max_results: int = 10) -> List[Tuple]:
        """객체 참조자 찾기."""
        results = []
        
        for obj in gc.get_objects():
            if type(obj).__name__ == obj_type:
                referrers = gc.get_referrers(obj)
                results.append((obj, referrers))
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def get_unreachable(self) -> List[Tuple[str, int]]:
        """도달 불가능한 객체 조회."""
        gc.collect()
        garbage = gc.garbage
        
        type_counts = {}
        for obj in garbage:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        return sorted(type_counts.items(), key=lambda x: x[1], reverse=True)


class AdaptiveGCTuner:
    """적응형 GC 튜너 - 메모리 사용량에 따라 자동 조정."""
    
    def __init__(self, base_config: GCConfig):
        self.base_config = base_config
        self.tuner = GCTuner(base_config)
        self._lock = RLock()
        self._memory_samples: List[Tuple[float, float]] = []
        self._last_strategy = None
        self._last_change_time = 0.0 # 마지막 변경 시간 추적
    
    def adapt_to_memory_pressure(self, memory_usage_percent: float):
        """메모리 압력에 따라 GC 조정."""
        import time
        current_time = time.time()
        
        with self._lock:
            self._memory_samples.append((current_time, memory_usage_percent))
            if len(self._memory_samples) > 100:
                self._memory_samples.pop(0)
        
        # 최소 30초 동안은 상태 변경을 보류 (잦은 변경 방지)
        if current_time - self._last_change_time < 30.0 and self._last_strategy is not None:
            return

        # 메모리 증가 추세 계산
        if len(self._memory_samples) > 10:
            recent_avg = sum(p for _, p in self._memory_samples[-10:]) / 10
            older_avg = sum(p for _, p in self._memory_samples[-20:-10]) / 10
            growth_rate = recent_avg - older_avg
        else:
            growth_rate = 0.0
        
        # 전략 결정 (임계값에 2% 정도의 마진을 주어 진동 방지)
        current_strategy = None
        if memory_usage_percent > 92: # 90 -> 92
            current_strategy = "aggressive"
        elif memory_usage_percent > 77 or growth_rate > 5: # 75 -> 77
            current_strategy = "balanced"
        elif memory_usage_percent < 48 and growth_rate < 2: # 50 -> 48
            current_strategy = "lazy"

        # 전략이 실제로 바뀌었을 때만 적용
        if current_strategy and current_strategy != self._last_strategy:
            if current_strategy == "aggressive":
                logger.warning(f"고메모리 상태 ({memory_usage_percent:.1f}%), 적극 GC 적용")
                self.tuner.set_thresholds(300, 5, 5)
            elif current_strategy == "balanced":
                logger.info(f"메모리 부하 감지, 균형 GC 모드 전환")
                self.tuner.set_thresholds(700, 10, 10)
            elif current_strategy == "lazy":
                logger.debug(f"여유 메모리 상태, 지연 GC 모드 유지") # INFO -> DEBUG
                self.tuner.set_thresholds(1000, 20, 20)
            
            self._last_strategy = current_strategy
            self._last_change_time = current_time
            if current_strategy == "aggressive":
                self.tuner.force_collect()


class ContextualGCManager:
    """컨텍스트 기반 GC 관리."""
    
    def __init__(self):
        self.tuner = GCTuner(GCConfig())
        self._lock = RLock()
        self._context_stack: List[str] = []
    
    def enter_performance_critical_section(self):
        """성능 중요 구간 진입 (GC 비활성화)."""
        with self._lock:
            if not self._context_stack:
                gc.disable()
                logger.debug("성능 중요 구간 시작: GC 비활성화")
            self._context_stack.append("perf_critical")
    
    def exit_performance_critical_section(self):
        """성능 중요 구간 종료 (GC 활성화)."""
        with self._lock:
            if self._context_stack:
                self._context_stack.pop()
            
            if not self._context_stack:
                gc.enable()
                self.tuner.force_collect()
                logger.debug("성능 중요 구간 종료: GC 재활성화 및 강제 수집")
    
    def enter_batch_processing(self):
        """배치 처리 구간 진입 (임계값 상향)."""
        with self._lock:
            self.tuner.set_thresholds(1500, 30, 30)
            logger.debug("배치 처리 시작: GC 임계값 상향")
            self._context_stack.append("batch_processing")
    
    def exit_batch_processing(self):
        """배치 처리 구간 종료."""
        with self._lock:
            if self._context_stack:
                self._context_stack.pop()
            
            if not self._context_stack:
                self.tuner.set_thresholds(700, 10, 10)  # 기본값으로 복구
                self.tuner.force_collect()
                logger.debug("배치 처리 종료: GC 설정 복구 및 강제 수집")


# 전역 인스턴스
_tuner_instance: Optional[GCTuner] = None
_adaptive_tuner_instance: Optional[AdaptiveGCTuner] = None
_contextual_manager_instance: Optional[ContextualGCManager] = None


def get_gc_tuner(config: Optional[GCConfig] = None) -> GCTuner:
    """전역 GC 튜너 조회."""
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = GCTuner(config or GCConfig())
    return _tuner_instance


def get_adaptive_gc_tuner(base_config: Optional[GCConfig] = None) -> AdaptiveGCTuner:
    """전역 적응형 GC 튜너 조회."""
    global _adaptive_tuner_instance
    if _adaptive_tuner_instance is None:
        _adaptive_tuner_instance = AdaptiveGCTuner(base_config or GCConfig())
    return _adaptive_tuner_instance


def get_contextual_gc_manager() -> ContextualGCManager:
    """전역 컨텍스트 기반 GC 관리자 조회."""
    global _contextual_manager_instance
    if _contextual_manager_instance is None:
        _contextual_manager_instance = ContextualGCManager()
    return _contextual_manager_instance


def reset_gc_tuner():
    """GC 튜너 리셋 (테스트용)."""
    global _tuner_instance
    _tuner_instance = None


def reset_adaptive_gc_tuner():
    """적응형 GC 튜너 리셋 (테스트용)."""
    global _adaptive_tuner_instance
    _adaptive_tuner_instance = None


def reset_contextual_gc_manager():
    """컨텍스트 기반 GC 관리자 리셋 (테스트용)."""
    global _contextual_manager_instance
    _contextual_manager_instance = None
