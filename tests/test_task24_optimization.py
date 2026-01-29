"""
Test Suite for Task 24: Performance Optimization System
"""

import time
import tempfile
import shutil
from pathlib import Path
from src.cache.caching_layer import (
    MultiLayerCache,
    MemoryCache,
    DiskCache,
    EvictionPolicy,
    CacheLevel,
)
from src.core.query_optimizer import QueryOptimizer, JoinStrategy
from src.services.monitoring.performance_monitor_advanced import (
    AdvancedPerformanceMonitor,
)


class TestMemoryCache:
    """Test in-memory L1 cache"""

    def test_01_put_and_get(self):
        """Test basic put/get operations"""
        cache = MemoryCache(max_size=100)

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_02_cache_miss(self):
        """Test cache miss"""
        cache = MemoryCache(max_size=100)

        result = cache.get("nonexistent")

        assert result is None

    def test_03_ttl_expiration(self):
        """Test TTL expiration"""
        cache = MemoryCache(max_size=100)

        cache.put("key1", "value1", ttl_seconds=1)
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_04_lru_eviction(self):
        """Test LRU eviction"""
        cache = MemoryCache(max_size=3, eviction_policy=EvictionPolicy.LRU)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    def test_05_cache_stats(self):
        """Test cache statistics"""
        cache = MemoryCache(max_size=100)

        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats.total_hits == 1
        assert stats.total_misses == 1

    def test_06_delete_entry(self):
        """Test deleting entry"""
        cache = MemoryCache(max_size=100)

        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_07_clear_cache(self):
        """Test clearing cache"""
        cache = MemoryCache(max_size=100)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cleared = cache.clear()

        assert cleared == 2
        assert cache.get("key1") is None


class TestDiskCache:
    """Test disk-based L2 cache"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DiskCache(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_08_put_and_get(self):
        """Test disk cache put/get"""
        self.cache.put("key1", {"data": "value1"})
        result = self.cache.get("key1")

        assert result == {"data": "value1"}

    def test_09_disk_cache_miss(self):
        """Test disk cache miss"""
        result = self.cache.get("nonexistent")

        assert result is None

    def test_10_disk_cache_ttl(self):
        """Test disk cache TTL"""
        self.cache.put("key1", "value1", ttl_seconds=1)
        assert self.cache.get("key1") == "value1"

        time.sleep(1.1)
        assert self.cache.get("key1") is None


class TestMultiLayerCache:
    """Test multi-layer caching system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLayerCache(l1_max_size=10, l2_cache_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_11_multi_layer_put_get(self):
        """Test multi-layer put/get"""
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")

        assert result == "value1"

    def test_12_l2_fallback(self):
        """Test L2 fallback when L1 miss"""
        self.cache.put("key1", "value1", cache_levels=[CacheLevel.L2])
        result = self.cache.get("key1")

        assert result == "value1"

    def test_13_cache_warmup(self):
        """Test cache warmup"""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}

        count = self.cache.warmup(data)

        assert count == 3
        assert self.cache.get("key1") == "value1"

    def test_14_multi_layer_delete(self):
        """Test deleting from multi-layer cache"""
        self.cache.put("key1", "value1")
        self.cache.delete("key1")

        assert self.cache.get("key1") is None

    def test_15_multi_layer_clear(self):
        """Test clearing multi-layer cache"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")

        result = self.cache.clear()

        assert result["l1"] > 0 or result["l2"] > 0


class TestQueryOptimizer:
    """Test query optimization"""

    def test_16_query_optimization(self):
        """Test query optimization"""
        optimizer = QueryOptimizer()

        query = {
            "type": "search",
            "keyword": "test",
            "where": {"field": "value"},
        }

        plan = optimizer.optimize(query)

        assert plan.query_id is not None
        assert len(plan.steps) > 0

    def test_17_query_plan_caching(self):
        """Test query plan caching"""
        optimizer = QueryOptimizer()

        query = {"type": "search", "keyword": "test"}

        plan1 = optimizer.optimize(query)
        plan2 = optimizer.optimize(query)

        assert plan1.query_id == plan2.query_id

    def test_18_index_suggestions(self):
        """Test index suggestions"""
        optimizer = QueryOptimizer()

        query = {
            "where": {"user_id": 123},
            "order_by": {"timestamp": "desc"},
        }

        suggestions = optimizer.index_manager.suggest_indexes(query)

        assert len(suggestions) > 0

    def test_19_join_strategy_selection(self):
        """Test join strategy selection"""
        optimizer = QueryOptimizer()

        query = {
            "join": [{"table": "users", "on": "user_id"}],
            "limit": 1000,
        }

        plan = optimizer.optimize(query)

        assert plan.join_strategy == JoinStrategy.HASH_JOIN

    def test_20_query_statistics(self):
        """Test query statistics collection"""
        optimizer = QueryOptimizer()

        query = {"type": "search", "keyword": "test"}

        def mock_execution(q):
            return [{"id": 1, "text": "result"}]

        result, stats = optimizer.execute_and_track(query, mock_execution)

        assert stats.execution_time_ms >= 0
        assert stats.rows_returned > 0

    def test_21_slow_queries_detection(self):
        """Test detecting slow queries"""
        optimizer = QueryOptimizer()

        # Record a slow query
        query = {"type": "search", "keyword": "test"}

        def mock_slow_execution(q):
            time.sleep(0.2)  # 200ms
            return [{"id": 1}]

        optimizer.execute_and_track(query, mock_slow_execution)

        slow_queries = optimizer.get_slow_queries(threshold_ms=100)

        assert len(slow_queries) > 0

    def test_22_optimization_statistics(self):
        """Test optimization statistics"""
        optimizer = QueryOptimizer()

        query = {"type": "search", "keyword": "test"}

        def mock_execution(q):
            return [{"id": 1}]

        optimizer.execute_and_track(query, mock_execution)

        stats = optimizer.get_optimization_statistics()

        assert stats["total_queries"] >= 1
        assert stats["average_execution_time_ms"] >= 0


class TestPerformanceMonitoring:
    """Test performance monitoring"""

    def test_23_record_metric(self):
        """Test recording metrics"""
        monitor = AdvancedPerformanceMonitor()

        result = monitor.record_metric(
            "query_latency",
            100.0,
            "ms",
            component="search",
        )

        assert result is True

    def test_24_component_summary(self):
        """Test component performance summary"""
        monitor = AdvancedPerformanceMonitor()

        monitor.record_metric("query_latency", 100.0, "ms", "search")
        monitor.record_metric("query_latency", 150.0, "ms", "search")
        monitor.record_metric("query_latency", 80.0, "ms", "search")

        summary = monitor.get_component_summary("search")

        assert summary["min"] == 80.0
        assert summary["max"] == 150.0
        assert summary["mean"] > 0

    def test_25_bottleneck_detection(self):
        """Test bottleneck detection"""
        monitor = AdvancedPerformanceMonitor()

        # Record high query latency
        monitor.record_metric("query_latency", 1000.0, "ms", "search")

        report = monitor.get_bottleneck_report()

        assert len(report) > 0

    def test_26_system_health_status(self):
        """Test system health assessment"""
        monitor = AdvancedPerformanceMonitor()

        monitor.record_metric("query_latency", 100.0, "ms", "search")

        health = monitor.get_system_health()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "critical"]

    def test_27_performance_trend(self):
        """Test performance trend analysis"""
        monitor = AdvancedPerformanceMonitor()

        for i in range(5):
            monitor.record_metric("query_latency", 100.0 - i * 10, "ms", "search")

        trend = monitor.get_performance_trend("query_latency", minutes=60)

        assert trend["sample_count"] >= 5
        assert "trend" in trend

    def test_28_critical_bottleneck_alert(self):
        """Test critical bottleneck alert"""
        monitor = AdvancedPerformanceMonitor()

        # Record critical metric
        monitor.record_metric("query_latency", 2000.0, "ms", "search")

        report = monitor.get_bottleneck_report()

        assert any(r["severity"] == "critical" for r in report)


class TestCacheInvalidation:
    """Test cache invalidation"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLayerCache(l1_max_size=10, l2_cache_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_29_cache_invalidation_pattern(self):
        """Test cache invalidation by pattern"""
        from src.cache.caching_layer import CacheInvalidator

        # Add entries
        self.cache.put("user_1_profile", {"name": "User 1"})
        self.cache.put("user_1_settings", {"theme": "dark"})
        self.cache.put("user_2_profile", {"name": "User 2"})

        invalidator = CacheInvalidator(self.cache)
        invalidator.register_pattern("user_1", ["user_1_profile", "user_1_settings"])

        count = invalidator.invalidate_pattern("user_1")

        assert count == 2
        assert self.cache.get("user_1_profile") is None
        assert self.cache.get("user_2_profile") is not None


class TestIntegratedOptimization:
    """Integrated optimization tests"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_30_full_optimization_pipeline(self):
        """Test full optimization pipeline"""
        # Setup components
        cache = MultiLayerCache(l1_max_size=100, l2_cache_dir=self.temp_dir)
        optimizer = QueryOptimizer()
        monitor = AdvancedPerformanceMonitor()

        # Warm cache
        cache.warmup(
            {
                "user_1": {"name": "User 1"},
                "user_2": {"name": "User 2"},
            }
        )

        # Execute optimized query
        query = {"type": "search", "keyword": "test"}

        def execution_func(q):
            value = cache.get("user_1")
            return [value] if value else []

        result, stats = optimizer.execute_and_track(query, execution_func)

        # Record metrics
        monitor.record_metric("query_latency", stats.execution_time_ms, "ms")

        # Verify results
        health = monitor.get_system_health()
        assert health["status"] in ["healthy", "degraded", "critical"]

    def test_31_end_to_end_performance_optimization(self):
        """Test end-to-end performance optimization"""
        cache = MultiLayerCache(l1_max_size=100, l2_cache_dir=self.temp_dir)
        optimizer = QueryOptimizer()
        monitor = AdvancedPerformanceMonitor()

        # Multiple queries
        queries = [
            {"type": "search", "keyword": "query1"},
            {"type": "search", "keyword": "query2"},
            {"type": "search", "keyword": "query3"},
        ]

        for query in queries:

            def execution_func(q):
                return [{"result": "data"}]

            result, stats = optimizer.execute_and_track(query, execution_func)
            monitor.record_metric("query_latency", stats.execution_time_ms, "ms")

        # Get statistics
        opt_stats = optimizer.get_optimization_statistics()
        health = monitor.get_system_health()

        assert opt_stats["total_queries"] == 3
        assert health["metrics_collected"] >= 3
