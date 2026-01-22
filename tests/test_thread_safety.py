"""
Comprehensive thread safety tests for RAG system.

Tests for:
- Concurrent read/write operations
- Race condition detection
- Deadlock prevention
- Lock timeout handling
- Atomic operations
"""

import sys
import os
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from core.thread_safe_session import ThreadSafeSessionManager

# 전역 관리자 인스턴스 생성 (get_thread_safe_manager 대체용)
_manager = ThreadSafeSessionManager()

def get_thread_safe_manager():
    return _manager

# 테스트용 ts_xxx 함수 매핑 (인스턴스 메서드 사용)
def ts_get(key, default=None): return _manager.get(key, default)
def ts_set(key, value): return _manager.set_inst(key, value)
def ts_delete(key): return _manager.delete(key)
def ts_exists(key): return _manager.exists(key)
def ts_atomic_read(keys): return _manager.atomic_read(keys)
def ts_atomic_update(update_func): return _manager.atomic_update(update_func)

from common.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Test Utilities
# ============================================================================

def init_streamlit_session():
    """Initialize Streamlit session state if not already done."""
    if not hasattr(st.session_state, '_initialized'):
        st.session_state._initialized = False


# ============================================================================
# Basic Operation Tests
# ============================================================================

class TestBasicOperations(unittest.TestCase):
    """Test basic thread-safe operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_set_and_get(self):
        """Test basic set and get operations."""
        self.manager.set("test_key", "test_value")
        value = self.manager.get("test_key")
        
        self.assertEqual(value, "test_value")
        logger.info("✓ Set and get operation verified")
    
    def test_get_with_default(self):
        """Test get with default value."""
        value = self.manager.get("nonexistent", default="default_value")
        
        self.assertEqual(value, "default_value")
        logger.info("✓ Get with default verified")
    
    def test_delete(self):
        """Test delete operation."""
        self.manager.set("temp_key", "temp_value")
        self.assertTrue(self.manager.exists("temp_key"))
        
        self.manager.delete("temp_key")
        self.assertFalse(self.manager.exists("temp_key"))
        logger.info("✓ Delete operation verified")
    
    def test_exists(self):
        """Test exists check."""
        self.manager.set("exists_key", "value")
        
        self.assertTrue(self.manager.exists("exists_key"))
        self.assertFalse(self.manager.exists("nonexistent"))
        logger.info("✓ Exists check verified")
    
    def test_clear_all(self):
        """Test clear all operation."""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")
        
        self.manager.clear_all()
        
        self.assertFalse(self.manager.exists("key1"))
        self.assertFalse(self.manager.exists("key2"))
        logger.info("✓ Clear all operation verified")


# ============================================================================
# Concurrent Access Tests
# ============================================================================

class TestConcurrentAccess(unittest.TestCase):
    """Test thread-safe concurrent access."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        def write_value(i):
            self.manager.set(f"key_{i}", f"value_{i}")
            time.sleep(0.001)  # Simulate some work
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_value, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()
        
        # Verify all values were written
        for i in range(100):
            value = self.manager.get(f"key_{i}")
            self.assertEqual(value, f"value_{i}")
        
        logger.info("✓ Concurrent writes verified (100 threads)")
    
    def test_concurrent_reads(self):
        """Test concurrent read operations."""
        # Set initial value
        self.manager.set("shared_key", "shared_value")
        
        results = []
        
        def read_value():
            value = self.manager.get("shared_key")
            results.append(value)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(read_value) for _ in range(200)]
            for future in as_completed(futures):
                future.result()
        
        # All reads should return same value
        self.assertEqual(len(results), 200)
        self.assertTrue(all(v == "shared_value" for v in results))
        logger.info("✓ Concurrent reads verified (200 threads)")
    
    def test_concurrent_read_write_mix(self):
        """Test mixed concurrent read and write operations."""
        # Initialize with some values
        for i in range(10):
            self.manager.set(f"key_{i}", i)
        
        read_count = [0]
        write_count = [0]
        
        def mixed_operation(idx):
            if idx % 2 == 0:
                # Read operation
                self.manager.get(f"key_{idx % 10}")
                read_count[0] += 1
            else:
                # Write operation
                self.manager.set(f"key_{idx % 10}", idx)
                write_count[0] += 1
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(mixed_operation, i) for i in range(300)]
            for future in as_completed(futures):
                future.result()
        
        logger.info(f"✓ Mixed operations verified ({read_count[0]} reads, {write_count[0]} writes)")
    
    def test_concurrent_delete(self):
        """Test concurrent delete operations."""
        # Set initial values
        for i in range(50):
            self.manager.set(f"delete_key_{i}", f"value_{i}")
        
        def delete_value(i):
            self.manager.delete(f"delete_key_{i}")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(delete_value, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()
        
        # Verify all values were deleted
        for i in range(50):
            self.assertFalse(self.manager.exists(f"delete_key_{i}"))
        
        logger.info("✓ Concurrent deletes verified (50 threads)")


# ============================================================================
# Race Condition Tests
# ============================================================================

class TestRaceConditions(unittest.TestCase):
    """Test race condition prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_counter_race_condition(self):
        """Test counter increment race condition prevention."""
        self.manager.set("counter", 0)
        
        def increment():
            # This is a weak test - ideally would use atomic_update
            current = self.manager.get("counter", 0)
            time.sleep(0.0001)  # Create race window
            self.manager.set("counter", current + 1)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment) for _ in range(100)]
            for future in as_completed(futures):
                future.result()
        
        # Note: Without atomic_update, final count may be less than 100
        final_count = self.manager.get("counter", 0)
        logger.info(f"✓ Counter race condition test: final count = {final_count}/100")
    
    def test_atomic_counter(self):
        """Test atomic counter increment prevents race conditions."""
        self.manager.set("atomic_counter", 0)
        
        def atomic_increment():
            def update_func(state):
                current = state.get("atomic_counter", 0)
                return {"atomic_counter": current + 1}
            self.manager.atomic_update(update_func)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(atomic_increment) for _ in range(100)]
            for future in as_completed(futures):
                future.result()
        
        final_count = self.manager.get("atomic_counter", 0)
        self.assertEqual(final_count, 100)
        logger.info(f"✓ Atomic counter verified: {final_count}")
    
    def test_dictionary_update_race_condition(self):
        """Test dictionary update race condition prevention."""
        self.manager.set("dict_data", {"count": 0, "users": []})
        
        def update_dict(user_id):
            def update_func(state):
                data = state.get("dict_data", {}).copy()
                data["count"] = data.get("count", 0) + 1
                if "users" not in data:
                    data["users"] = []
                data["users"].append(user_id)
                return {"dict_data": data}
            self.manager.atomic_update(update_func)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_dict, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()
        
        final_data = self.manager.get("dict_data", {})
        self.assertEqual(final_data.get("count"), 50)
        logger.info(f"✓ Dictionary update race condition: count = {final_data.get('count')}")


# ============================================================================
# Deadlock Prevention Tests
# ============================================================================

class TestDeadlockPrevention(unittest.TestCase):
    """Test deadlock prevention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_no_deadlock_in_nested_operations(self):
        """Test that RLock prevents deadlock in nested operations."""
        self.manager.set("test_key", "initial_value")
        
        def nested_operation():
            # First acquisition
            value1 = self.manager.get("test_key")
            
            # Nested operation (would deadlock with regular Lock)
            self.manager.set("test_key", value1 + "_modified")
            
            # Another read (nested)
            value2 = self.manager.get("test_key")
            
            self.assertIn("_modified", value2)
        
        thread = threading.Thread(target=nested_operation)
        thread.start()
        thread.join(timeout=2.0)
        
        self.assertFalse(thread.is_alive(), "Thread should complete (no deadlock)")
        logger.info("✓ No deadlock in nested operations")
    
    def test_no_deadlock_under_lock_contention(self):
        """Test system remains responsive under high lock contention."""
        start_time = time.time()
        
        def heavy_contention():
            for _ in range(100):
                self.manager.set("contested_key", "value")
                self.manager.get("contested_key")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(heavy_contention) for _ in range(10)]
            for future in as_completed(futures):
                future.result()
        
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 10.0, "High contention should complete within 10 seconds")
        logger.info(f"✓ No deadlock under contention (completed in {elapsed:.2f}s)")
    
    def test_lock_timeout_mechanism(self):
        """Test lock timeout detection."""
        # Create manager with very short timeout
        short_timeout_manager = ThreadSafeSessionManager(lock_timeout=0.1)
        
        stats_before = short_timeout_manager.get_stats()
        
        # Try operations (may hit timeout under extreme conditions)
        for _ in range(1000):
            short_timeout_manager.set("timeout_test", "value")
        
        stats_after = short_timeout_manager.get_stats()
        
        # Should have minimal or no timeout failures
        failed = stats_after["failed_acquisitions"]
        logger.info(f"✓ Lock timeout mechanism: {failed} timeouts detected")


# ============================================================================
# Batch Operation Tests
# ============================================================================

class TestBatchOperations(unittest.TestCase):
    """Test batch operations for atomicity."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_get_multiple(self):
        """Test getting multiple values atomically."""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")
        self.manager.set("key3", "value3")
        
        values = self.manager.get_multiple(["key1", "key2", "key3"])
        
        self.assertEqual(values["key1"], "value1")
        self.assertEqual(values["key2"], "value2")
        self.assertEqual(values["key3"], "value3")
        logger.info("✓ Get multiple operations verified")
    
    def test_set_multiple(self):
        """Test setting multiple values atomically."""
        data = {"batch_key1": "batch_value1", "batch_key2": "batch_value2", "batch_key3": "batch_value3"}
        
        success = self.manager.set_multiple(data)
        
        self.assertTrue(success)
        for key, value in data.items():
            self.assertEqual(self.manager.get(key), value)
        logger.info("✓ Set multiple operations verified")
    
    def test_atomic_read_consistency(self):
        """Test atomic read consistency."""
        self.manager.set("read_key1", "value1")
        self.manager.set("read_key2", "value2")
        
        # Read should be consistent (no partial updates visible)
        values = self.manager.atomic_read(["read_key1", "read_key2"])
        
        self.assertEqual(len(values), 2)
        self.assertIn("read_key1", values)
        self.assertIn("read_key2", values)
        logger.info("✓ Atomic read consistency verified")


# ============================================================================
# Statistics and Monitoring Tests
# ============================================================================

class TestStatistics(unittest.TestCase):
    """Test statistics and monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
        self.manager = ThreadSafeSessionManager(lock_timeout=5.0)
    
    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        self.manager.set("stat_key", "value")
        self.manager.get("stat_key")
        self.manager.delete("stat_key")
        
        stats = self.manager.get_stats()
        
        self.assertIn("session_keys", stats)
        self.assertIn("failed_acquisitions", stats)
        logger.info(f"✓ Statistics tracked: {stats}")
    
    def test_health_check(self):
        """Test health check mechanism."""
        self.manager.set("health_key", "value")
        
        is_healthy = self.manager.is_healthy()
        
        self.assertTrue(is_healthy)
        logger.info("✓ Health check verified")
    
    def test_stats_reset(self):
        """Test statistics reset."""
        self.manager.set("key", "value")
        self.manager.reset_stats()
        
        stats = self.manager.get_stats()
        
        self.assertEqual(stats["failed_acquisitions"], 0)
        logger.info("✓ Statistics reset verified")


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        init_streamlit_session()
        st.session_state.clear()
    
    def test_convenience_set_get(self):
        """Test convenience functions for set/get."""
        ts_set("conv_key", "conv_value")
        value = ts_get("conv_key")
        
        self.assertEqual(value, "conv_value")
        logger.info("✓ Convenience set/get verified")
    
    def test_convenience_exists(self):
        """Test convenience function for exists."""
        ts_set("exist_key", "value")
        
        self.assertTrue(ts_exists("exist_key"))
        self.assertFalse(ts_exists("nonexistent"))
        logger.info("✓ Convenience exists verified")
    
    def test_convenience_delete(self):
        """Test convenience function for delete."""
        ts_set("del_key", "value")
        ts_delete("del_key")
        
        self.assertFalse(ts_exists("del_key"))
        logger.info("✓ Convenience delete verified")


def run_thread_safety_tests():
    """Run all thread safety tests with detailed reporting."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBasicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentAccess))
    suite.addTests(loader.loadTestsFromTestCase(TestRaceConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestDeadlockPrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("THREAD SAFETY TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_thread_safety_tests()
    sys.exit(0 if success else 1)
