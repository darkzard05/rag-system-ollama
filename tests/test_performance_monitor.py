"""
Comprehensive tests for performance monitoring system.

Tests for:
- Response time tracking
- Memory monitoring
- Token counting
- Operation metrics
- Report generation
"""

import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.monitoring.performance_monitor import (
    ResponseTimeTracker,
    MemoryMonitor,
    TokenCounter,
    PerformanceMonitor,
    OperationType,
    OperationMetrics,
)
from common.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Response Time Tracker Tests
# ============================================================================

class TestResponseTimeTracker(unittest.TestCase):
    """Test response time tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ResponseTimeTracker(max_history=100)
    
    def test_record_single_duration(self):
        """Test recording single duration."""
        self.tracker.record_duration(OperationType.LLM_INFERENCE, 1.5)
        stats = self.tracker.get_stats(OperationType.LLM_INFERENCE)
        
        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['avg'], 1.5)
        self.assertEqual(stats['min'], 1.5)
        self.assertEqual(stats['max'], 1.5)
        logger.info("✓ Single duration recording verified")
    
    def test_record_multiple_durations(self):
        """Test recording multiple durations."""
        durations = [0.5, 1.0, 1.5, 2.0, 2.5]
        for duration in durations:
            self.tracker.record_duration(OperationType.LLM_INFERENCE, duration)
        
        stats = self.tracker.get_stats(OperationType.LLM_INFERENCE)
        
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['min'], 0.5)
        self.assertEqual(stats['max'], 2.5)
        self.assertEqual(stats['avg'], 1.5)
        logger.info("✓ Multiple duration recording verified")
    
    def test_percentile_calculation(self):
        """Test percentile calculations."""
        for i in range(1, 101):
            self.tracker.record_duration(OperationType.LLM_INFERENCE, i * 0.01)
        
        stats = self.tracker.get_stats(OperationType.LLM_INFERENCE)
        
        self.assertIsNotNone(stats['p50'])
        self.assertIsNotNone(stats['p95'])
        self.assertIsNotNone(stats['p99'])
        self.assertLess(stats['p50'], stats['p95'])
        self.assertLess(stats['p95'], stats['p99'])
        logger.info(f"✓ Percentiles verified: p50={stats['p50']:.3f}, p95={stats['p95']:.3f}, p99={stats['p99']:.3f}")
    
    def test_multiple_operation_types(self):
        """Test tracking different operation types."""
        self.tracker.record_duration(OperationType.LLM_INFERENCE, 1.0)
        self.tracker.record_duration(OperationType.DOCUMENT_RETRIEVAL, 0.5)
        self.tracker.record_duration(OperationType.EMBEDDING_GENERATION, 0.2)
        
        llm_stats = self.tracker.get_stats(OperationType.LLM_INFERENCE)
        retrieval_stats = self.tracker.get_stats(OperationType.DOCUMENT_RETRIEVAL)
        embedding_stats = self.tracker.get_stats(OperationType.EMBEDDING_GENERATION)
        
        self.assertEqual(llm_stats['count'], 1)
        self.assertEqual(retrieval_stats['count'], 1)
        self.assertEqual(embedding_stats['count'], 1)
        logger.info("✓ Multiple operation types tracking verified")
    
    def test_clear_timings(self):
        """Test clearing timings."""
        self.tracker.record_duration(OperationType.LLM_INFERENCE, 1.0)
        self.tracker.clear()
        
        stats = self.tracker.get_stats(OperationType.LLM_INFERENCE)
        self.assertEqual(stats['count'], 0)
        logger.info("✓ Clear operation verified")


# ============================================================================
# Memory Monitor Tests
# ============================================================================

class TestMemoryMonitor(unittest.TestCase):
    """Test memory monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = MemoryMonitor(max_history=100)
    
    def test_get_current_usage(self):
        """Test reading current memory usage."""
        memory = self.monitor.get_current_usage()
        
        self.assertIn('rss_mb', memory)
        self.assertIn('vms_mb', memory)
        self.assertGreater(memory['rss_mb'], 0)
        self.assertGreater(memory['vms_mb'], 0)
        logger.info(f"✓ Memory usage: RSS={memory['rss_mb']:.2f}MB, VMS={memory['vms_mb']:.2f}MB")
    
    def test_memory_delta_calculation(self):
        """Test memory delta calculation."""
        start = {"rss_mb": 100.0, "vms_mb": 200.0}
        end = {"rss_mb": 105.0, "vms_mb": 210.0}
        
        delta = self.monitor.get_memory_delta(start, end)
        
        self.assertEqual(delta['rss_delta_mb'], 5.0)
        self.assertEqual(delta['vms_delta_mb'], 10.0)
        logger.info(f"✓ Memory delta: RSS={delta['rss_delta_mb']:.2f}MB, VMS={delta['vms_delta_mb']:.2f}MB")
    
    def test_memory_samples_collection(self):
        """Test memory sample collection."""
        for _ in range(10):
            self.monitor.get_current_usage()
            time.sleep(0.01)
        
        stats = self.monitor.get_stats()
        
        self.assertEqual(stats['sample_count'], 10)
        self.assertGreaterEqual(stats['max_rss_mb'], stats['min_rss_mb'])
        logger.info(f"✓ Memory samples collected: {stats['sample_count']}")
    
    def test_clear_samples(self):
        """Test clearing memory samples."""
        self.monitor.get_current_usage()
        self.monitor.clear()
        
        stats = self.monitor.get_stats()
        self.assertEqual(stats['sample_count'], 0)
        logger.info("✓ Memory samples cleared")


# ============================================================================
# Token Counter Tests
# ============================================================================

class TestTokenCounter(unittest.TestCase):
    """Test token counting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
    
    def test_empty_text(self):
        """Test counting tokens in empty text."""
        tokens = TokenCounter.count_tokens("")
        self.assertEqual(tokens, 0)
        logger.info("✓ Empty text token counting verified")
    
    def test_simple_text(self):
        """Test token counting on simple text."""
        text = "Hello world, this is a test."
        tokens = TokenCounter.count_tokens(text)
        
        # Rough estimation: ~4 chars per token
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))
        logger.info(f"✓ Simple text tokens: {len(text)} chars → {tokens} tokens")
    
    def test_long_text(self):
        """Test token counting on longer text."""
        text = " ".join(["word"] * 1000)
        tokens = TokenCounter.count_tokens(text)
        
        self.assertGreater(tokens, 0)
        logger.info(f"✓ Long text tokens: {len(text)} chars → {tokens} tokens")
    
    def test_count_tokens_in_list(self):
        """Test counting tokens in list of texts."""
        texts = ["Hello world", "This is a test", "Multiple documents"]
        total_tokens = TokenCounter.count_tokens_in_list(texts)
        
        individual_sum = sum(TokenCounter.count_tokens(t) for t in texts)
        self.assertEqual(total_tokens, individual_sum)
        logger.info(f"✓ List token counting: {len(texts)} texts → {total_tokens} tokens")


# ============================================================================
# Performance Monitor Tests
# ============================================================================

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(enable_memory_tracking=True)
    
    def test_context_manager_basic(self):
        """Test operation tracking with context manager."""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
            time.sleep(0.05)
            op.tokens = 100
        
        # Check that operation was recorded
        count = self.monitor.get_operation_count(OperationType.LLM_INFERENCE)
        self.assertEqual(count, 1)
        logger.info("✓ Context manager operation tracking verified")
    
    def test_operation_metrics(self):
        """Test operation metrics recording."""
        with self.monitor.track_operation(OperationType.EMBEDDING_GENERATION, {"model": "e5"}) as op:
            time.sleep(0.05)
            op.tokens = 50
        
        stats = self.monitor.get_operation_stats(OperationType.EMBEDDING_GENERATION)
        
        self.assertEqual(stats.successful_operations, 1)
        self.assertGreater(stats.avg_duration_seconds, 0)
        self.assertEqual(stats.total_tokens, 50)
        logger.info(f"✓ Operation metrics: duration={stats.avg_duration_seconds:.3f}s, tokens={stats.total_tokens}")
    
    def test_error_tracking(self):
        """Test error tracking in operations."""
        try:
            with self.monitor.track_operation(OperationType.QUERY_PROCESSING):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        stats = self.monitor.get_operation_stats(OperationType.QUERY_PROCESSING)
        
        self.assertEqual(stats.failed_operations, 1)
        logger.info("✓ Error tracking verified")
    
    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        for i in range(5):
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                time.sleep(0.01)
                op.tokens = (i + 1) * 10
        
        count = self.monitor.get_operation_count(OperationType.LLM_INFERENCE)
        self.assertEqual(count, 5)
        
        stats = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)
        self.assertEqual(stats.total_tokens, sum(range(1, 6)) * 10)
        logger.info(f"✓ Multiple operations: {count} tracked, {stats.total_tokens} total tokens")
    
    def test_operation_type_filtering(self):
        """Test filtering operations by type."""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE):
            pass
        
        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL):
            pass
        
        with self.monitor.track_operation(OperationType.EMBEDDING_GENERATION):
            pass
        
        llm_count = self.monitor.get_operation_count(OperationType.LLM_INFERENCE)
        retrieval_count = self.monitor.get_operation_count(OperationType.DOCUMENT_RETRIEVAL)
        embedding_count = self.monitor.get_operation_count(OperationType.EMBEDDING_GENERATION)
        
        self.assertEqual(llm_count, 1)
        self.assertEqual(retrieval_count, 1)
        self.assertEqual(embedding_count, 1)
        logger.info("✓ Operation type filtering verified")
    
    def test_get_all_stats(self):
        """Test getting all statistics."""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE):
            pass
        
        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL):
            pass
        
        all_stats = self.monitor.get_all_stats()
        
        self.assertIn(OperationType.LLM_INFERENCE, all_stats)
        self.assertIn(OperationType.DOCUMENT_RETRIEVAL, all_stats)
        logger.info(f"✓ All stats retrieved: {len(all_stats)} operation types")
    
    def test_memory_stats(self):
        """Test memory statistics."""
        memory_stats = self.monitor.get_memory_stats()
        
        self.assertIn('current_rss_mb', memory_stats)
        self.assertGreaterEqual(memory_stats['current_rss_mb'], 0)
        logger.info(f"✓ Memory stats: current={memory_stats['current_rss_mb']:.2f}MB")
    
    def test_health_status(self):
        """Test health status check."""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE):
            pass
        
        health = self.monitor.get_health_status()
        
        self.assertIn('status', health)
        self.assertIn('memory_mb', health)
        self.assertIn('total_operations', health)
        self.assertIn('issues', health)
        logger.info(f"✓ Health status: {health['status']}, memory={health['memory_mb']:.2f}MB")


# ============================================================================
# Report Generation Tests
# ============================================================================

class TestReportGeneration(unittest.TestCase):
    """Test report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(enable_memory_tracking=True)
    
    def test_empty_report(self):
        """Test report generation with no operations."""
        report = self.monitor.generate_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('total_operations', report)
        self.assertIn('memory', report)
        self.assertIn('operations', report)
        self.assertEqual(report['total_operations'], 0)
        logger.info("✓ Empty report generation verified")
    
    def test_report_with_operations(self):
        """Test report generation with operations."""
        for i in range(3):
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                time.sleep(0.01)
                op.tokens = 100
        
        report = self.monitor.generate_report()
        
        self.assertEqual(report['total_operations'], 3)
        self.assertIn('llm_inference', report['operations'])
        
        llm_op = report['operations']['llm_inference']
        self.assertEqual(llm_op['total'], 3)
        self.assertEqual(llm_op['tokens']['total'], 300)
        logger.info(f"✓ Report with operations: {report['total_operations']} operations logged")
    
    def test_report_structure(self):
        """Test report structure completeness."""
        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL):
            time.sleep(0.01)
        
        report = self.monitor.generate_report()
        
        # Check structure
        self.assertIsInstance(report['memory'], dict)
        self.assertIsInstance(report['operations'], dict)
        
        if 'document_retrieval' in report['operations']:
            op_data = report['operations']['document_retrieval']
            self.assertIn('duration', op_data)
            self.assertIn('tokens', op_data)
            self.assertIn('memory_delta_mb', op_data)
        
        logger.info("✓ Report structure verified")
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE):
            pass
        
        self.monitor.clear_metrics()
        
        count = self.monitor.get_operation_count()
        self.assertEqual(count, 0)
        logger.info("✓ Metrics clearing verified")


# ============================================================================
# Integration Tests
# ============================================================================

class TestMonitoringIntegration(unittest.TestCase):
    """Integration tests for performance monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(enable_memory_tracking=True)
    
    def test_full_rag_simulation(self):
        """Test monitoring a simulated RAG pipeline."""
        # Document retrieval
        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL, {"docs": 10}) as op:
            time.sleep(0.01)
            op.tokens = 500
        
        # Embedding generation
        with self.monitor.track_operation(OperationType.EMBEDDING_GENERATION, {"model": "e5"}) as op:
            time.sleep(0.02)
            op.tokens = 200
        
        # Reranking
        with self.monitor.track_operation(OperationType.RERANKING, {"top_k": 5}) as op:
            time.sleep(0.015)
            op.tokens = 100
        
        # LLM inference
        with self.monitor.track_operation(OperationType.LLM_INFERENCE, {"model": "qwen"}) as op:
            time.sleep(0.05)
            op.tokens = 300
        
        # Query processing (overall)
        with self.monitor.track_operation(OperationType.QUERY_PROCESSING):
            pass
        
        # Verify all operations tracked
        total = self.monitor.get_operation_count()
        self.assertEqual(total, 5)
        
        report = self.monitor.generate_report()
        self.assertEqual(report['total_operations'], 5)
        
        logger.info(f"✓ Full RAG simulation: {total} operations tracked")
    
    def test_concurrent_operation_tracking(self):
        """Test tracking multiple concurrent-like operations."""
        import threading
        
        def do_operation(op_type):
            with self.monitor.track_operation(op_type) as op:
                time.sleep(0.01)
                op.tokens = 100
        
        threads = []
        for op_type in [OperationType.LLM_INFERENCE, OperationType.DOCUMENT_RETRIEVAL, OperationType.EMBEDDING_GENERATION]:
            t = threading.Thread(target=do_operation, args=(op_type,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        count = self.monitor.get_operation_count()
        self.assertEqual(count, 3)
        logger.info(f"✓ Concurrent operations: {count} operations tracked")
    
    def test_performance_degradation_detection(self):
        """Test detecting performance degradation."""
        # Record fast operations
        for i in range(5):
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                time.sleep(0.01)
                op.tokens = 100
        
        stats_fast = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)
        
        # Record slower operations
        for i in range(5):
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                time.sleep(0.05)
                op.tokens = 100
        
        stats_slow = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)
        
        # New average should be higher
        self.assertGreater(stats_slow.avg_duration_seconds, stats_fast.avg_duration_seconds)
        logger.info(f"✓ Degradation detected: {stats_fast.avg_duration_seconds:.3f}s → {stats_slow.avg_duration_seconds:.3f}s")


def run_performance_monitoring_tests():
    """Run all performance monitoring tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestResponseTimeTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenCounter))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitoringIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE MONITORING TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_monitoring_tests()
    sys.exit(0 if success else 1)
