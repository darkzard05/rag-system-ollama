"""
Performance monitoring system for RAG pipeline.

Tracks response times, memory usage, token counts, and other metrics
for comprehensive performance analysis and optimization.

Features:
- Response time tracking with histogram stats
- Memory usage monitoring (RSS, VMS)
- Token counting for LLM operations
- Performance aggregation and reporting
- Real-time metrics collection
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import json

try:
    from common.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class OperationType(Enum):
    """Types of operations to monitor."""
    DOCUMENT_RETRIEVAL = "document_retrieval"
    EMBEDDING_GENERATION = "embedding_generation"
    RERANKING = "reranking"
    LLM_INFERENCE = "llm_inference"
    QUERY_PROCESSING = "query_processing"
    PDF_LOADING = "pdf_loading"
    SEMANTIC_CHUNKING = "semantic_chunking"


class MetricType(Enum):
    """Types of metrics to track."""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    TOKEN_COUNT = "token_count"
    ERROR_COUNT = "error_count"
    THROUGHPUT = "throughput"


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_type: OperationType
    start_time: float
    end_time: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    tokens: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Get memory delta in MB."""
        if self.memory_start is None or self.memory_end is None:
            return None
        # memory_start/end are recorded in MB (see MemoryMonitor.get_current_usage and OperationTracker)
        return self.memory_end - self.memory_start
    
    @property
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.end_time is not None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    operation_type: OperationType
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    min_duration_seconds: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    avg_duration_seconds: float = 0.0
    p50_duration_seconds: float = 0.0
    p95_duration_seconds: float = 0.0
    p99_duration_seconds: float = 0.0
    total_memory_delta_mb: float = 0.0
    avg_memory_delta_mb: float = 0.0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Response Time Tracker
# ============================================================================

class ResponseTimeTracker:
    """Track response times for operations."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize response time tracker.
        
        Args:
            max_history: Maximum number of samples to keep
        """
        self._lock = threading.RLock()
        self._max_history = max_history
        self._timings: Dict[OperationType, deque] = {}
        logger.info(f"ResponseTimeTracker initialized (max_history={max_history})")
    
    def record_duration(self, operation_type: OperationType, duration_seconds: float) -> None:
        """
        Record operation duration.
        
        Args:
            operation_type: Type of operation
            duration_seconds: Duration in seconds
        """
        with self._lock:
            if operation_type not in self._timings:
                self._timings[operation_type] = deque(maxlen=self._max_history)
            
            self._timings[operation_type].append(duration_seconds)
            logger.debug(f"[{operation_type.value}] Duration recorded: {duration_seconds:.3f}s")
    
    def get_stats(self, operation_type: OperationType) -> Dict[str, float]:
        """
        Get timing statistics for operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Dictionary with min, max, avg, p50, p95, p99 values
        """
        with self._lock:
            if operation_type not in self._timings or len(self._timings[operation_type]) == 0:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0
                }
            
            timings = sorted(self._timings[operation_type])
            count = len(timings)
            
            return {
                "count": count,
                "min": float(min(timings)),
                "max": float(max(timings)),
                "avg": sum(timings) / count,
                "p50": self._percentile(timings, 50),
                "p95": self._percentile(timings, 95),
                "p99": self._percentile(timings, 99),
                "total": sum(timings)
            }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        index = int((len(data) - 1) * percentile / 100)
        return float(data[index])
    
    def clear(self) -> None:
        """Clear all timings."""
        with self._lock:
            self._timings.clear()
            logger.info("Response time tracker cleared")


# ============================================================================
# Memory Monitor
# ============================================================================

class MemoryMonitor:
    """Monitor memory usage."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize memory monitor.
        
        Args:
            max_history: Maximum number of samples to keep
        """
        self._lock = threading.RLock()
        self._max_history = max_history
        self._memory_samples: deque = deque(maxlen=max_history)
        self._process = psutil.Process()
        logger.info(f"MemoryMonitor initialized (max_history={max_history})")
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current memory usage in MB.
        
        Returns:
            Dictionary with RSS and VMS in MB
        """
        try:
            with self._lock:
                mem_info = self._process.memory_info()
                sample = {
                    "timestamp": datetime.now(),
                    "rss_mb": mem_info.rss / (1024 * 1024),
                    "vms_mb": mem_info.vms / (1024 * 1024),
                }
                self._memory_samples.append(sample)
                
                return {
                    "rss_mb": sample["rss_mb"],
                    "vms_mb": sample["vms_mb"]
                }
        except Exception as e:
            logger.error(f"Error reading memory usage: {e}")
            return {"rss_mb": 0.0, "vms_mb": 0.0}
    
    def get_memory_delta(self, start_memory: Dict[str, float], end_memory: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate memory delta between two measurements.
        
        Args:
            start_memory: Starting memory measurement
            end_memory: Ending memory measurement
            
        Returns:
            Dictionary with RSS and VMS delta in MB
        """
        return {
            "rss_delta_mb": end_memory["rss_mb"] - start_memory["rss_mb"],
            "vms_delta_mb": end_memory["vms_mb"] - start_memory["vms_mb"],
        }
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get memory statistics from samples.
        
        Returns:
            Dictionary with min, max, avg memory usage
        """
        with self._lock:
            if len(self._memory_samples) == 0:
                return {
                    "min_rss_mb": 0.0,
                    "max_rss_mb": 0.0,
                    "avg_rss_mb": 0.0,
                    "current_rss_mb": 0.0,
                    "sample_count": 0
                }
            
            rss_values = [s["rss_mb"] for s in self._memory_samples]
            
            return {
                "min_rss_mb": min(rss_values),
                "max_rss_mb": max(rss_values),
                "avg_rss_mb": sum(rss_values) / len(rss_values),
                "current_rss_mb": rss_values[-1],
                "sample_count": len(rss_values)
            }
    
    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._memory_samples.clear()
            logger.info("Memory monitor cleared")


# ============================================================================
# Token Counter
# ============================================================================

class TokenCounter:
    """Count tokens in text (simple implementation)."""
    
    def __init__(self):
        """Initialize token counter."""
        logger.info("TokenCounter initialized")
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Simple token count estimation.
        
        For accurate token counting with specific models,
        integrate with tiktoken or model-specific tokenizers.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token (common average)
        # For production, use proper tokenizer
        if not text:
            return 0
        
        # Split by whitespace and estimate
        words = text.split()
        # Average word length ~5 chars, average ~1.3 tokens per word
        token_estimate = int(len(text) / 4)
        return max(1, token_estimate)
    
    @staticmethod
    def count_tokens_in_list(texts: List[str]) -> int:
        """
        Count tokens in multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Total token count
        """
        return sum(TokenCounter.count_tokens(text) for text in texts)


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
        """
        self._lock = threading.RLock()
        self._enable_memory = enable_memory_tracking
        self._response_tracker = ResponseTimeTracker()
        self._memory_monitor = MemoryMonitor()
        self._token_counter = TokenCounter()
        self._operations: List[OperationMetrics] = []
        self._max_operations = 10000
        logger.info(f"PerformanceMonitor initialized (memory_tracking={enable_memory_tracking})")
    
    # ========================================================================
    # Context Manager Support for Tracking
    # ========================================================================
    
    def track_operation(self, operation_type: OperationType, metadata: Optional[Dict[str, Any]] = None):
        """
        Create operation context manager for tracking.
        
        Usage:
            with monitor.track_operation(OperationType.LLM_INFERENCE, {"model": "qwen"}) as op:
                # do work
                op.tokens = 150
        
        Args:
            operation_type: Type of operation
            metadata: Optional metadata dictionary
            
        Returns:
            Context manager for operation tracking
        """
        return OperationTracker(self, operation_type, metadata or {})
    
    def record_operation(self, metrics: OperationMetrics) -> None:
        """
        Record completed operation metrics.
        
        Args:
            metrics: Operation metrics to record
        """
        with self._lock:
            if len(self._operations) >= self._max_operations:
                # Keep recent operations
                self._operations = self._operations[-5000:]
            
            self._operations.append(metrics)
            
            # Record timing
            if metrics.duration_seconds is not None:
                self._response_tracker.record_duration(
                    metrics.operation_type,
                    metrics.duration_seconds
                )
            
            logger.debug(
                f"[{metrics.operation_type.value}] Operation recorded: "
                f"duration={metrics.duration_seconds:.3f}s, "
                f"tokens={metrics.tokens}, "
                f"error={metrics.error}"
            )
    
    # ========================================================================
    # Manual Recording
    # ========================================================================
    
    def record_response_time(self, operation_type: OperationType, duration_seconds: float) -> None:
        """
        Manually record response time.
        
        Args:
            operation_type: Type of operation
            duration_seconds: Duration in seconds
        """
        self._response_tracker.record_duration(operation_type, duration_seconds)
    
    def record_token_count(self, operation_type: OperationType, token_count: int) -> None:
        """
        Record token count for operation.
        
        Args:
            operation_type: Type of operation
            token_count: Number of tokens
        """
        with self._lock:
            logger.debug(f"[{operation_type.value}] Tokens recorded: {token_count}")
    
    # ========================================================================
    # Statistics Retrieval
    # ========================================================================
    
    def get_operation_stats(self, operation_type: OperationType) -> PerformanceStats:
        """
        Get aggregated statistics for operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            PerformanceStats object
        """
        with self._lock:
            # Filter operations of this type
            ops = [o for o in self._operations if o.operation_type == operation_type]
            
            if not ops:
                return PerformanceStats(operation_type=operation_type)
            
            # Calculate statistics
            completed_ops = [o for o in ops if o.is_completed]
            failed_ops = [o for o in ops if o.error is not None]
            successful_ops = [o for o in completed_ops if o.error is None]
            
            durations = [o.duration_seconds for o in successful_ops if o.duration_seconds is not None]
            
            # Percentiles
            sorted_durations = sorted(durations) if durations else []
            
            def percentile(data: List[float], p: int) -> float:
                if not data:
                    return 0.0
                idx = int(len(data) * p / 100)
                return float(data[min(idx, len(data) - 1)])
            
            # Memory stats
            memory_deltas = [o.memory_delta_mb for o in completed_ops if o.memory_delta_mb is not None]
            
            # Token stats
            token_counts = [o.tokens for o in completed_ops if o.tokens > 0]
            
            return PerformanceStats(
                operation_type=operation_type,
                total_operations=len(ops),
                successful_operations=len(successful_ops),
                failed_operations=len(failed_ops),
                total_duration_seconds=sum(durations),
                min_duration_seconds=min(durations) if durations else None,
                max_duration_seconds=max(durations) if durations else None,
                avg_duration_seconds=sum(durations) / len(durations) if durations else 0.0,
                p50_duration_seconds=percentile(sorted_durations, 50),
                p95_duration_seconds=percentile(sorted_durations, 95),
                p99_duration_seconds=percentile(sorted_durations, 99),
                total_memory_delta_mb=sum(memory_deltas),
                avg_memory_delta_mb=sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0.0,
                total_tokens=sum(token_counts),
                avg_tokens=sum(token_counts) / len(token_counts) if token_counts else 0.0,
            )
    
    def get_all_stats(self) -> Dict[OperationType, PerformanceStats]:
        """
        Get statistics for all operation types.
        
        Returns:
            Dictionary mapping operation types to statistics
        """
        with self._lock:
            return {op_type: self.get_operation_stats(op_type) for op_type in OperationType}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        return self._memory_monitor.get_stats()
    
    # ========================================================================
    # Reporting
    # ========================================================================
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance data
        """
        with self._lock:
            all_stats = self.get_all_stats()
            memory_stats = self.get_memory_stats()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_operations": len(self._operations),
                "memory": memory_stats,
                "operations": {}
            }
            
            # Add operation-specific stats
            for op_type, stats in all_stats.items():
                if stats.total_operations > 0:
                    report["operations"][op_type.value] = {
                        "total": stats.total_operations,
                        "successful": stats.successful_operations,
                        "failed": stats.failed_operations,
                        "duration": {
                            "min": round(stats.min_duration_seconds or 0, 4),
                            "max": round(stats.max_duration_seconds or 0, 4),
                            "avg": round(stats.avg_duration_seconds, 4),
                            "p50": round(stats.p50_duration_seconds, 4),
                            "p95": round(stats.p95_duration_seconds, 4),
                            "p99": round(stats.p99_duration_seconds, 4),
                        },
                        "tokens": {
                            "total": stats.total_tokens,
                            "avg": round(stats.avg_tokens, 2)
                        },
                        "memory_delta_mb": {
                            "total": round(stats.total_memory_delta_mb, 2),
                            "avg": round(stats.avg_memory_delta_mb, 2)
                        }
                    }
            
            return report
    
    def print_report(self) -> None:
        """Print formatted performance report."""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("PERFORMANCE MONITORING REPORT")
        print("=" * 70)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Operations: {report['total_operations']}")
        print()
        
        # Memory stats
        print("Memory Usage:")
        print(f"  Current RSS: {report['memory'].get('current_rss_mb', 0):.2f} MB")
        print(f"  Min RSS: {report['memory'].get('min_rss_mb', 0):.2f} MB")
        print(f"  Max RSS: {report['memory'].get('max_rss_mb', 0):.2f} MB")
        print(f"  Avg RSS: {report['memory'].get('avg_rss_mb', 0):.2f} MB")
        print()
        
        # Operation stats
        print("Operation Performance:")
        for op_type, op_stats in report['operations'].items():
            print(f"\n  {op_type}:")
            print(f"    Total: {op_stats['total']} | Success: {op_stats['successful']} | Failed: {op_stats['failed']}")
            print(f"    Duration (s): min={op_stats['duration']['min']:.3f}, "
                  f"avg={op_stats['duration']['avg']:.3f}, "
                  f"p95={op_stats['duration']['p95']:.3f}, "
                  f"p99={op_stats['duration']['p99']:.3f}")
            print(f"    Tokens: total={op_stats['tokens']['total']}, avg={op_stats['tokens']['avg']:.1f}")
            print(f"    Memory Δ: avg={op_stats['memory_delta_mb']['avg']:.2f} MB")
        
        print("\n" + "=" * 70)
    
    def export_metrics_json(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to export JSON file
        """
        try:
            report = self.generate_report()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    # ========================================================================
    # Management
    # ========================================================================
    
    def get_operation_count(self, operation_type: Optional[OperationType] = None) -> int:
        """
        Get count of operations.
        
        Args:
            operation_type: Filter by type (None for all)
            
        Returns:
            Number of operations
        """
        with self._lock:
            if operation_type is None:
                return len(self._operations)
            return sum(1 for o in self._operations if o.operation_type == operation_type)
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._operations.clear()
            self._response_tracker.clear()
            self._memory_monitor.clear()
            logger.info("All metrics cleared")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status based on performance metrics.
        
        Returns:
            Dictionary with health information
        """
        with self._lock:
            memory_stats = self.get_memory_stats()
            all_stats = self.get_all_stats()
            
            # Check for issues
            issues = []
            
            # Check memory
            current_rss = memory_stats.get('current_rss_mb', 0)
            if current_rss > 1000:  # Threshold: 1GB
                issues.append(f"High memory usage: {current_rss:.0f} MB")
            
            # Check error rates
            for op_type, stats in all_stats.items():
                if stats.total_operations > 0:
                    error_rate = stats.failed_operations / stats.total_operations
                    if error_rate > 0.1:  # Threshold: 10% error rate
                        issues.append(f"High error rate in {op_type.value}: {error_rate:.1%}")
            
            return {
                "status": "healthy" if not issues else "warning",
                "memory_mb": current_rss,
                "total_operations": len(self._operations),
                "issues": issues
            }


# ============================================================================
# Operation Tracker Context Manager
# ============================================================================

class OperationTracker:
    """Context manager for tracking individual operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_type: OperationType, metadata: Dict[str, Any]):
        """
        Initialize operation tracker.
        
        Args:
            monitor: PerformanceMonitor instance
            operation_type: Type of operation
            metadata: Metadata dictionary
        """
        self.monitor = monitor
        self.operation_type = operation_type
        self.metadata = metadata
        self.metrics = OperationMetrics(
            operation_type=operation_type,
            start_time=time.time(),
            metadata=metadata
        )
        self.tokens = 0
    
    def __enter__(self):
        """Enter context manager."""
        if self.monitor._enable_memory:
            self.metrics.memory_start = self.monitor._memory_monitor.get_current_usage()["rss_mb"]
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.metrics.end_time = time.time()
        
        if self.monitor._enable_memory:
            current = self.monitor._memory_monitor.get_current_usage()
            self.metrics.memory_end = current["rss_mb"]
        
        self.metrics.tokens = self.tokens
        
        if exc_type is not None:
            self.metrics.error = f"{exc_type.__name__}: {exc_val}"
        
        self.monitor.record_operation(self.metrics)


# ============================================================================
# Global Instance
# ============================================================================

_global_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get or create global performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor(enable_memory_tracking=True)
                logger.info("Global PerformanceMonitor created")
    
    return _global_monitor


# ============================================================================
# Convenience Functions
# ============================================================================

def track_operation(operation_type: OperationType, metadata: Optional[Dict[str, Any]] = None):
    """
    Track an operation using global monitor.
    
    Usage:
        with track_operation(OperationType.LLM_INFERENCE) as op:
            # do work
            op.tokens = 150
    
    Args:
        operation_type: Type of operation
        metadata: Optional metadata
        
    Returns:
        Context manager
    """
    return get_performance_monitor().track_operation(operation_type, metadata)


def record_response_time(operation_type: OperationType, duration_seconds: float) -> None:
    """Record response time to global monitor."""
    get_performance_monitor().record_response_time(operation_type, duration_seconds)


def get_performance_report() -> Dict[str, Any]:
    """Get performance report from global monitor."""
    return get_performance_monitor().generate_report()


def print_performance_report() -> None:
    """Print performance report."""
    get_performance_monitor().print_report()
