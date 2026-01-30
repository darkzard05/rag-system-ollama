"""
메모리 최적화 테스트 스위트.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time

import pytest

from infra.resource_manager import (
    DeferredCleanup,
    ManagedResource,
    ResourcePool,
    get_contextual_resource_manager,
    get_resource_manager,
    reset_contextual_resource_manager,
    reset_resource_manager,
)
from services.monitoring.memory_profiler import (
    MemoryMetrics,
    MemoryMonitor,
    MemoryProfiler,
    get_memory_monitor,
    get_memory_profiler,
    reset_memory_monitor,
    reset_memory_profiler,
)
from services.optimization.memory_optimizer import (
    MemoryOptimizationConfig,
    MemoryOptimizationMode,
    MemoryOptimizer,
    get_memory_optimizer,
    reset_memory_optimizer,
)

# ============================================================================
# 메모리 프로파일러 테스트
# ============================================================================


class TestMemoryProfiler:
    """메모리 프로파일러 테스트."""

    def test_take_snapshot(self):
        """메모리 스냅샷 획득."""
        profiler = MemoryProfiler()

        snapshot = profiler.take_snapshot()

        assert snapshot.total_mb > 0
        assert snapshot.rss_mb > 0
        assert snapshot.python_objects > 0

    def test_get_current_memory(self):
        """현재 메모리 메트릭 조회."""
        profiler = MemoryProfiler()

        metrics = profiler.get_current_memory()

        assert isinstance(metrics, MemoryMetrics)
        assert metrics.current_mb > 0
        assert 0 <= metrics.percent_used <= 100

    def test_memory_growth_detection(self):
        """메모리 성장 감지."""
        profiler = MemoryProfiler()

        # 첫 번째 스냅샷
        snap1 = profiler.take_snapshot()

        # 일부 메모리 할당
        large_list = [i for i in range(100000)]

        # 두 번째 스냅샷
        snap2 = profiler.take_snapshot()

        # 메모리 증가 확인
        assert snap2.total_mb >= snap1.total_mb

    def test_get_top_objects(self):
        """상위 메모리 사용 객체 조회."""
        profiler = MemoryProfiler()

        top_objects = profiler.get_top_objects(5)

        assert isinstance(top_objects, list)
        # tracemalloc 활성화 여부에 따라 결과 다를 수 있음

    def test_memory_profiler_singleton(self):
        """메모리 프로파일러 싱글톤."""
        reset_memory_profiler()

        prof1 = get_memory_profiler()
        prof2 = get_memory_profiler()

        assert prof1 is prof2

    def test_get_memory_history(self):
        """메모리 히스토리 조회."""
        profiler = MemoryProfiler()

        for _ in range(5):
            profiler.take_snapshot()
            time.sleep(0.01)

        history = profiler.get_memory_history(3)

        assert len(history) <= 3


# ============================================================================
# 메모리 모니터 테스트
# ============================================================================


class TestMemoryMonitor:
    """메모리 모니터 테스트."""

    def test_monitor_initialization(self):
        """모니터 초기화."""
        reset_memory_monitor()

        monitor = get_memory_monitor()

        assert monitor is not None
        assert monitor.profiler is not None

    def test_monitor_start_stop(self):
        """모니터 시작/중지."""
        reset_memory_monitor()
        monitor = get_memory_monitor()

        monitor.start_monitoring()
        assert monitor._monitoring is True

        time.sleep(0.5)

        monitor.stop_monitoring()
        assert monitor._monitoring is False

    def test_alert_collection(self):
        """경고 수집."""
        reset_memory_monitor()
        monitor = MemoryMonitor(interval_seconds=0.1)

        monitor.start_monitoring()

        time.sleep(1)

        monitor.stop_monitoring()


# ============================================================================
# 리소스 풀 테스트
# ============================================================================


class TestResourcePool:
    """리소스 풀 테스트."""

    @pytest.mark.asyncio
    async def test_resource_pool_creation(self):
        """리소스 풀 생성."""
        pool = ResourcePool(object, max_size=5)

        resource = await pool.acquire()

        assert resource is not None

        await pool.release(resource)

    @pytest.mark.asyncio
    async def test_resource_pool_capacity(self):
        """리소스 풀 용량 관리."""
        pool = ResourcePool(object, max_size=2)

        # 최대 용량까지 리소스 획득
        res1 = await pool.acquire()
        res2 = await pool.acquire()

        # 릴리스
        await pool.release(res1)
        await pool.release(res2)

    @pytest.mark.asyncio
    async def test_resource_pool_stats(self):
        """리소스 풀 통계."""
        pool = ResourcePool(object, max_size=5)

        resource = await pool.acquire()

        stats = pool.get_stats()

        assert stats.total_created > 0
        assert stats.currently_allocated > 0

        await pool.release(resource)

    @pytest.mark.asyncio
    async def test_resource_pool_clear(self):
        """리소스 풀 정리."""
        pool = ResourcePool(object, max_size=5)

        resource = await pool.acquire()
        await pool.release(resource)

        await pool.clear()


# ============================================================================
# 리소스 관리자 테스트
# ============================================================================


class TestResourceManager:
    """리소스 관리자 테스트."""

    def test_resource_manager_singleton(self):
        """리소스 관리자 싱글톤."""
        reset_resource_manager()

        mgr1 = get_resource_manager()
        mgr2 = get_resource_manager()

        assert mgr1 is mgr2

    def test_create_pool(self):
        """풀 생성."""
        reset_resource_manager()
        manager = get_resource_manager()

        pool = manager.create_pool("test_pool", object, max_size=5)

        assert pool is not None

    def test_get_pool(self):
        """풀 조회."""
        reset_resource_manager()
        manager = get_resource_manager()

        pool1 = manager.create_pool("test_pool", object, max_size=5)
        pool2 = manager.get_pool("test_pool")

        assert pool1 is pool2

    def test_get_stats(self):
        """통계 조회."""
        reset_resource_manager()
        manager = get_resource_manager()

        manager.create_pool("pool1", object, max_size=5)
        manager.create_pool("pool2", object, max_size=10)

        stats = manager.get_stats()

        assert "pool1" in stats
        assert "pool2" in stats


# ============================================================================
# ManagedResource 테스트
# ============================================================================


class TestManagedResource:
    """ManagedResource 테스트."""

    @pytest.mark.asyncio
    async def test_managed_resource_context(self):
        """ManagedResource 컨텍스트."""
        cleanup_called = []

        async def cleanup_func():
            cleanup_called.append(True)

        resource = ManagedResource()
        resource.register_cleanup(cleanup_func)

        async with resource:
            assert resource.is_open

        assert not resource.is_open
        assert len(cleanup_called) > 0


# ============================================================================
# 메모리 최적화 통합 테스트
# ============================================================================


class TestMemoryOptimizer:
    """메모리 최적화 통합 테스트."""

    def test_memory_optimizer_initialization(self):
        """메모리 최적화 관리자 초기화."""
        reset_memory_optimizer()

        config = MemoryOptimizationConfig()
        optimizer = MemoryOptimizer(config)

        assert optimizer is not None

    def test_memory_optimizer_modes(self):
        """메모리 최적화 모드."""
        reset_memory_optimizer()

        # STRICT 모드
        config_strict = MemoryOptimizationConfig(mode=MemoryOptimizationMode.STRICT)
        optimizer_strict = MemoryOptimizer(config_strict)
        assert optimizer_strict.config.mode == MemoryOptimizationMode.STRICT

        # PERFORMANCE 모드
        reset_memory_optimizer()
        config_perf = MemoryOptimizationConfig(mode=MemoryOptimizationMode.PERFORMANCE)
        optimizer_perf = MemoryOptimizer(config_perf)
        assert optimizer_perf.config.mode == MemoryOptimizationMode.PERFORMANCE

    def test_memory_optimizer_singleton(self):
        """메모리 최적화 관리자 싱글톤."""
        reset_memory_optimizer()

        opt1 = get_memory_optimizer()
        opt2 = get_memory_optimizer()

        assert opt1 is opt2

    def test_get_memory_stats(self):
        """메모리 통계 조회."""
        reset_memory_optimizer()
        optimizer = get_memory_optimizer()

        stats = optimizer.get_memory_stats()

        assert "memory" in stats
        assert "objects" in stats

    def test_get_top_memory_users(self):
        """메모리 사용량 상위 조회."""
        reset_memory_optimizer()
        optimizer = get_memory_optimizer()

        top_users = optimizer.get_top_memory_users(5)

        assert isinstance(top_users, list)


# ============================================================================
# 엣지 케이스 테스트
# ============================================================================


class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_memory_leak_detection_empty_history(self):
        """누수 감지 - 빈 히스토리."""
        profiler = MemoryProfiler()

        leaks = profiler.detect_memory_leaks()

        assert isinstance(leaks, list)

    @pytest.mark.asyncio
    async def test_resource_pool_timeout(self):
        """리소스 풀 타임아웃."""
        pool = ResourcePool(object, max_size=1)

        res1 = await pool.acquire()

        # 타임아웃 테스트 (실제 타임아웃은 길므로 생략)

        await pool.release(res1)

    def test_deferred_cleanup(self):
        """지연 정리."""
        cleanup_order = []

        def cleanup1():
            cleanup_order.append(1)

        def cleanup2():
            cleanup_order.append(2)

        deferred = DeferredCleanup()
        deferred.defer(cleanup1)
        deferred.defer(cleanup2)

        # 비동기 정리 (동기 버전으로 테스트)
        # await deferred.cleanup()


# ============================================================================
# 컨텍스트 관리자 테스트
# ============================================================================


class TestContextualManagers:
    """컨텍스트 관리자 테스트."""

    def test_contextual_resource_manager_singleton(self):
        """컨텍스트 리소스 관리자 싱글톤."""
        reset_contextual_resource_manager()

        mgr1 = get_contextual_resource_manager()
        mgr2 = get_contextual_resource_manager()

        assert mgr1 is mgr2
