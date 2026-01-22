"""
메모리 프로파일링 및 분석 모듈 - 메모리 사용량 추적, 누수 감지.
"""

import gc
import sys
import tracemalloc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from threading import RLock, Thread
from collections import defaultdict
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryAlertLevel(Enum):
    """메모리 경고 수준."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MemorySnapshot:
    """메모리 스냅샷."""
    timestamp: float
    total_mb: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    python_objects: int
    tracemalloc_peak_mb: float = 0.0
    tracemalloc_current_mb: float = 0.0


@dataclass
class MemoryMetrics:
    """메모리 메트릭."""
    current_mb: float
    peak_mb: float
    available_mb: float
    percent_used: float
    num_objects: int
    num_tracked_objects: int = 0
    growth_rate_mb_per_sec: float = 0.0
    gc_collections: int = 0
    gc_unreachable: int = 0


@dataclass
class MemoryLeak:
    """메모리 누수 정보."""
    object_type: str
    count: int
    size_mb: float
    growth_rate: float
    first_seen: float
    last_seen: float


class MemoryProfiler:
    """메모리 프로파일러."""
    
    def __init__(self, enable_tracemalloc: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self._lock = RLock()
        self._snapshots: List[MemorySnapshot] = []
        self._leak_candidates: Dict[str, MemoryLeak] = {}
        self._object_history: Dict[str, List[int]] = defaultdict(list)
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Tracemalloc 활성화됨")
    
    def take_snapshot(self) -> MemorySnapshot:
        """현재 메모리 스냅샷 획득."""
        import time
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        total_mb = mem_info.rss / 1024 / 1024
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024
        
        # [최적화] gc.get_objects()는 매우 무거우므로 실시간 스냅샷에서는 제외
        num_objects = 0 
        
        tracemalloc_peak = 0.0
        tracemalloc_current = 0.0
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_peak = peak / 1024 / 1024
            tracemalloc_current = current / 1024 / 1024
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_mb=total_mb,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            python_objects=num_objects,
            tracemalloc_peak_mb=tracemalloc_peak,
            tracemalloc_current_mb=tracemalloc_current,
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            # 최근 1000개만 유지
            if len(self._snapshots) > 1000:
                self._snapshots.pop(0)
        
        return snapshot
    
    def get_current_memory(self) -> MemoryMetrics:
        """현재 메모리 메트릭 조회."""
        snapshot = self.take_snapshot()
        
        process = psutil.Process(os.getpid())
        virtual_mem = psutil.virtual_memory()
        
        available_mb = virtual_mem.available / 1024 / 1024
        percent_used = (snapshot.total_mb / (virtual_mem.total / 1024 / 1024)) * 100
        
        # 성장률 계산
        growth_rate = 0.0
        if len(self._snapshots) > 1:
            prev_snapshot = self._snapshots[-2]
            time_diff = snapshot.timestamp - prev_snapshot.timestamp
            if time_diff > 0:
                growth_rate = (snapshot.total_mb - prev_snapshot.total_mb) / time_diff
        
        gc_stats = gc.get_stats()
        gc_collections = sum(s.get("collections", 0) for s in gc_stats)
        gc_unreachable = sum(s.get("uncollectable", 0) for s in gc_stats)
        
        return MemoryMetrics(
            current_mb=snapshot.total_mb,
            peak_mb=max((s.total_mb for s in self._snapshots), default=0),
            available_mb=available_mb,
            percent_used=percent_used,
            num_objects=snapshot.python_objects,
            num_tracked_objects=0, # [최적화] gc.get_objects() 제거
            growth_rate_mb_per_sec=growth_rate,
            gc_collections=gc_collections,
            gc_unreachable=gc_unreachable,
        )
    
    def get_top_objects(self, top_n: int = 10) -> List[Tuple[str, int, float]]:
        """크기가 큰 상위 객체 조회.
        
        Returns:
            (객체 타입, 개수, 크기 MB) 튜플 리스트
        """
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        results = []
        for stat in top_stats[:top_n]:
            size_mb = stat.size / 1024 / 1024
            results.append((str(stat), stat.count, size_mb))
        
        return results
    
    def detect_memory_leaks(self, threshold_mb: float = 200.0) -> List[MemoryLeak]:
        """메모리 누수 감지.
        
        [개선]
        - 임계값 상향 (50MB -> 200MB)하여 대형 모델 로딩 수용.
        """
        if len(self._snapshots) < 5:
            return []
        
        with self._lock:
            # 최근 5개의 스냅샷 분석
            recent = self._snapshots[-5:]
            total_growth = recent[-1].total_mb - recent[0].total_mb
            
            # 지속적으로 증가하는지 확인
            is_persistently_growing = all(
                (recent[i].total_mb - recent[i-1].total_mb) > 0 
                for i in range(1, len(recent))
            )
            
            if is_persistently_growing and total_growth > threshold_mb:
                key = "PersistentGrowth"
                duration = recent[-1].timestamp - recent[0].timestamp
                growth_rate = total_growth / (duration / 60) # MB/min
                
                if growth_rate > threshold_mb: # 분당 200MB 이상 지속 증가 시
                    leak = MemoryLeak(
                        object_type=key,
                        count=len(recent),
                        size_mb=total_growth,
                        growth_rate=growth_rate,
                        first_seen=recent[0].timestamp,
                        last_seen=recent[-1].timestamp
                    )
                    return [leak]
            
            return []
    
    def get_memory_history(self, last_n: int = 100) -> List[MemorySnapshot]:
        """메모리 히스토리 조회."""
        with self._lock:
            return self._snapshots[-last_n:]
    
    def clear_history(self):
        """히스토리 초기화."""
        with self._lock:
            self._snapshots.clear()
            self._leak_candidates.clear()


class MemoryMonitor:
    """실시간 메모리 모니터."""
    
    def __init__(
        self,
        interval_seconds: float = 5.0,
        warning_threshold_percent: float = 80.0,
        critical_threshold_percent: float = 95.0,
    ):
        self.interval = interval_seconds
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        
        self.profiler = MemoryProfiler()
        self._lock = RLock()
        self._alerts: List[Tuple[float, MemoryAlertLevel, str]] = []
        self._monitoring = False
        self._monitor_thread: Optional[Thread] = None
    
    def start_monitoring(self):
        """모니터링 시작 (비활성화됨 - MemoryOptimizer 통합)"""
        # [최적화] 중복 실행 방지를 위해 실제 스레드 시작을 막음
        logger.debug("MemoryMonitor thread start requested, but skipped (Managed by MemoryOptimizer)")
        return
    
    def stop_monitoring(self):
        """모니터링 중지."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("메모리 모니터링 중지됨")
    
    def _monitor_loop(self):
        """모니터링 루프."""
        import time
        
        while self._monitoring:
            try:
                metrics = self.profiler.get_current_memory()
                
                if metrics.percent_used >= self.critical_threshold:
                    level = MemoryAlertLevel.CRITICAL
                    msg = f"CRITICAL: 메모리 사용량 {metrics.percent_used:.1f}%"
                elif metrics.percent_used >= self.warning_threshold:
                    level = MemoryAlertLevel.WARNING
                    msg = f"WARNING: 메모리 사용량 {metrics.percent_used:.1f}%"
                else:
                    level = MemoryAlertLevel.INFO
                    msg = f"INFO: 메모리 사용량 {metrics.percent_used:.1f}%"
                
                self._add_alert(level, msg)
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"모니터링 에러: {e}")
    
    def _add_alert(self, level: MemoryAlertLevel, message: str):
        """경고 추가."""
        import time
        
        with self._lock:
            self._alerts.append((time.time(), level, message))
            
            # 최근 1000개만 유지
            if len(self._alerts) > 1000:
                self._alerts.pop(0)
    
    def get_alerts(self, level: Optional[MemoryAlertLevel] = None) -> List[Tuple[float, str]]:
        """경고 조회."""
        with self._lock:
            if level is None:
                return [(t, m) for t, l, m in self._alerts]
            else:
                return [(t, m) for t, l, m in self._alerts if l == level]
    
    def clear_alerts(self):
        """경고 초기화."""
        with self._lock:
            self._alerts.clear()


# 전역 인스턴스
_profiler_instance: Optional[MemoryProfiler] = None
_monitor_instance: Optional[MemoryMonitor] = None


def get_memory_profiler() -> MemoryProfiler:
    """전역 메모리 프로파일러 조회."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = MemoryProfiler()
    return _profiler_instance


def get_memory_monitor() -> MemoryMonitor:
    """전역 메모리 모니터 조회."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = MemoryMonitor()
    return _monitor_instance


def reset_memory_profiler():
    """메모리 프로파일러 리셋 (테스트용)."""
    global _profiler_instance
    _profiler_instance = None


def reset_memory_monitor():
    """메모리 모니터 리셋 (테스트용)."""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop_monitoring()
    _monitor_instance = None
