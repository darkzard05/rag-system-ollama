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

import csv
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

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
    end_time: float | None = None
    memory_start: float | None = None
    memory_end: float | None = None
    tokens: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get operation duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def memory_delta_mb(self) -> float | None:
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
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
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
        self._lock = threading.RLock()
        self._max_history = max_history
        self._timings: dict[OperationType, deque] = {}

    def record_duration(
        self, operation_type: OperationType, duration_seconds: float
    ) -> None:
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
            logger.debug(
                f"[{operation_type.value}] Duration recorded: {duration_seconds:.3f}s"
            )

    def get_stats(self, operation_type: OperationType) -> dict[str, float]:
        """
        Get timing statistics for operation type.

        Args:
            operation_type: Type of operation

        Returns:
            Dictionary with min, max, avg, p50, p95, p99 values
        """
        with self._lock:
            if (
                operation_type not in self._timings
                or len(self._timings[operation_type]) == 0
            ):
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
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
                "total": sum(timings),
            }

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
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
        self._lock = threading.RLock()
        self._max_history = max_history
        self._memory_samples: deque = deque(maxlen=max_history)
        self._process = psutil.Process()

    def get_current_usage(self) -> dict[str, float]:
        """
        Get current memory usage in MB.

        Returns:
            Dictionary with RSS and VMS in MB
        """
        try:
            with self._lock:
                mem_info = self._process.memory_info()
                rss_mb = float(mem_info.rss / (1024 * 1024))
                vms_mb = float(mem_info.vms / (1024 * 1024))
                
                sample = {
                    "timestamp": datetime.now(),
                    "rss_mb": rss_mb,
                    "vms_mb": vms_mb,
                }
                self._memory_samples.append(sample)

                return {"rss_mb": rss_mb, "vms_mb": vms_mb}
        except Exception as e:
            logger.error(f"Error reading memory usage: {e}")
            return {"rss_mb": 0.0, "vms_mb": 0.0}

    def get_memory_delta(
        self, start_memory: dict[str, float], end_memory: dict[str, float]
    ) -> dict[str, float]:
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

    def get_stats(self) -> dict[str, float]:
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
                    "sample_count": 0,
                }

            rss_values = [s["rss_mb"] for s in self._memory_samples]

            return {
                "min_rss_mb": min(rss_values),
                "max_rss_mb": max(rss_values),
                "avg_rss_mb": sum(rss_values) / len(rss_values),
                "current_rss_mb": rss_values[-1],
                "sample_count": len(rss_values),
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
        logger.debug("[Monitor] [Init] TokenCounter initialized")

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
        text.split()
        # Average word length ~5 chars, average ~1.3 tokens per word
        token_estimate = int(len(text) / 4)
        return max(1, token_estimate)

    @staticmethod
    def count_tokens_in_list(texts: list[str]) -> int:
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

    def __init__(self, enable_memory_tracking: bool = False):
        self._lock = threading.RLock()
        self._enable_memory = enable_memory_tracking
        self._response_tracker = ResponseTimeTracker()
        self._memory_monitor = MemoryMonitor()
        self._token_counter = TokenCounter()
        self._operations: list[OperationMetrics] = []
        self._max_operations = 10000

        self.csv_path = os.path.join("logs", "performance_metrics.csv")
        self.jsonl_path = os.path.join(
            "logs", "eval", f"qa_history_{datetime.now().strftime('%Y%m')}.jsonl"
        )
        self._init_csv()
        self._init_jsonl()

        # [최적화] 비동기 로깅을 위한 큐와 스레드 설정
        self._log_queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self._log_thread.start()

        logger.info("[System] [Monitor] 성능 모니터링 시스템 활성화 (비동기 I/O)")

    def _init_csv(self):
        """CSV 파일 초기화 및 헤더 작성"""
        try:
            if not os.path.exists(os.path.dirname(self.csv_path)):
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Timestamp",
                            "Model",
                            "TTFT",
                            "Thinking_Time",
                            "Answer_Time",
                            "Total_Time",
                            "Tokens",
                            "TPS",
                            "Query",
                        ]
                    )
        except Exception as e:
            logger.error(f"Failed to initialize performance CSV: {e}")

    def _init_jsonl(self):
        """JSONL 폴더 초기화"""
        try:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize JSONL directory: {e}")

    def _logging_worker(self):
        """백그라운드에서 CSV 및 JSONL 기록을 처리하는 워커 스레드"""
        import json

        while not (self._stop_event.is_set() and self._log_queue.empty()):
            try:
                data_entry = self._log_queue.get(timeout=1.0)
                try:
                    # 데이터 타입에 따라 처리 (리스트면 CSV, 딕셔너리면 JSONL)
                    if isinstance(data_entry, list):
                        with open(
                            self.csv_path, "a", newline="", encoding="utf-8"
                        ) as f:
                            writer = csv.writer(f)
                            writer.writerow(data_entry)
                    elif isinstance(data_entry, dict):
                        with open(self.jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[Monitor] 로깅 쓰기 오류: {e}")
                finally:
                    self._log_queue.task_done()
            except Exception:
                continue

    def log_to_csv(self, data: dict[str, Any]):
        """성능 데이터를 큐에 삽입 (CSV용)"""
        try:
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("model", "unknown"),
                round(data.get("ttft", 0), 3),
                round(data.get("thinking", 0), 3),
                round(data.get("answer", 0), 3),
                round(data.get("total", 0), 3),
                data.get("tokens", 0),
                round(data.get("tps", 0), 2),
                data.get("query", "")[:100],
            ]
            self._log_queue.put(row)
        except Exception as e:
            logger.error(f"Failed to queue metrics for CSV: {e}")

    def log_qa_history(self, entry: dict[str, Any]):
        """QA 히스토리를 큐에 삽입 (JSONL용)"""
        try:
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.now().isoformat()
            self._log_queue.put(entry)
        except Exception as e:
            logger.error(f"Failed to queue QA history for JSONL: {e}")

    def stop(self):
        """모니터링 시스템 종료 및 남은 로그 플러시"""
        self._stop_event.set()
        if self._log_thread.is_alive():
            self._log_thread.join(timeout=2.0)

    # ========================================================================
    # Context Manager Support for Tracking
    # ========================================================================

    def track_operation(
        self, operation_type: OperationType, metadata: dict[str, Any] | None = None
    ):
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
                    metrics.operation_type, metrics.duration_seconds
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

    def record_response_time(
        self, operation_type: OperationType, duration_seconds: float
    ) -> None:
        """
        Manually record response time.

        Args:
            operation_type: Type of operation
            duration_seconds: Duration in seconds
        """
        self._response_tracker.record_duration(operation_type, duration_seconds)

    def record_token_count(
        self, operation_type: OperationType, token_count: int
    ) -> None:
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

            durations = [
                o.duration_seconds
                for o in successful_ops
                if o.duration_seconds is not None
            ]

            # Percentiles
            sorted_durations = sorted(durations) if durations else []

            def percentile(data: list[float], p: int) -> float:
                if not data:
                    return 0.0
                idx = int(len(data) * p / 100)
                return float(data[min(idx, len(data) - 1)])

            # Memory stats
            memory_deltas = [
                o.memory_delta_mb
                for o in completed_ops
                if o.memory_delta_mb is not None
            ]

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
                avg_duration_seconds=sum(durations) / len(durations)
                if durations
                else 0.0,
                p50_duration_seconds=percentile(sorted_durations, 50),
                p95_duration_seconds=percentile(sorted_durations, 95),
                p99_duration_seconds=percentile(sorted_durations, 99),
                total_memory_delta_mb=sum(memory_deltas),
                avg_memory_delta_mb=sum(memory_deltas) / len(memory_deltas)
                if memory_deltas
                else 0.0,
                total_tokens=sum(token_counts),
                avg_tokens=sum(token_counts) / len(token_counts)
                if token_counts
                else 0.0,
            )

    def get_all_stats(self) -> dict[OperationType, PerformanceStats]:
        """
        Get statistics for all operation types.

        Returns:
            Dictionary mapping operation types to statistics
        """
        with self._lock:
            return {
                op_type: self.get_operation_stats(op_type) for op_type in OperationType
            }

    def get_memory_stats(self) -> dict[str, float]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return self._memory_monitor.get_stats()

    # ========================================================================
    # Reporting
    # ========================================================================

    def generate_report(self) -> dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with performance data
        """
        with self._lock:
            all_stats = self.get_all_stats()
            memory_stats = self.get_memory_stats()

            report: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "total_operations": len(self._operations),
                "memory": memory_stats,
                "operations": {},
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
                            "avg": round(stats.avg_tokens, 2),
                        },
                        "memory_delta_mb": {
                            "total": round(stats.total_memory_delta_mb, 2),
                            "avg": round(stats.avg_memory_delta_mb, 2),
                        },
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
        for op_type, op_stats in report["operations"].items():
            print(f"\n  {op_type}:")
            print(
                f"    Total: {op_stats['total']} | Success: {op_stats['successful']} | Failed: {op_stats['failed']}"
            )
            print(
                f"    Duration (s): min={op_stats['duration']['min']:.3f}, "
                f"avg={op_stats['duration']['avg']:.3f}, "
                f"p95={op_stats['duration']['p95']:.3f}, "
                f"p99={op_stats['duration']['p99']:.3f}"
            )
            print(
                f"    Tokens: total={op_stats['tokens']['total']}, avg={op_stats['tokens']['avg']:.1f}"
            )
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
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    # ========================================================================
    # Management
    # ========================================================================

    def get_operation_count(self, operation_type: OperationType | None = None) -> int:
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
            return sum(
                1 for o in self._operations if o.operation_type == operation_type
            )

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._operations.clear()
            self._response_tracker.clear()
            self._memory_monitor.clear()
            logger.info("All metrics cleared")

    def get_health_status(self) -> dict[str, Any]:
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
            current_rss = memory_stats.get("current_rss_mb", 0)
            if current_rss > 1000:  # Threshold: 1GB
                issues.append(f"High memory usage: {current_rss:.0f} MB")

            # Check error rates
            for op_type, stats in all_stats.items():
                if stats.total_operations > 0:
                    error_rate = stats.failed_operations / stats.total_operations
                    if error_rate > 0.1:  # Threshold: 10% error rate
                        issues.append(
                            f"High error rate in {op_type.value}: {error_rate:.1%}"
                        )

            return {
                "status": "healthy" if not issues else "warning",
                "memory_mb": current_rss,
                "total_operations": len(self._operations),
                "issues": issues,
            }


# ============================================================================
# Operation Tracker Context Manager
# ============================================================================


class OperationTracker:
    """Context manager for tracking individual operations."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_type: OperationType,
        metadata: dict[str, Any],
    ):
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
            operation_type=operation_type, start_time=time.time(), metadata=metadata
        )
        self.tokens = 0

    def __enter__(self):
        """Enter context manager."""
        if self.monitor._enable_memory:
            self.metrics.memory_start = (
                self.monitor._memory_monitor.get_current_usage()["rss_mb"]
            )
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

_global_monitor: PerformanceMonitor | None = None
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
                _global_monitor = PerformanceMonitor(enable_memory_tracking=False)
                logger.debug("[Monitor] [Init] Global PerformanceMonitor created")

    return _global_monitor


# ============================================================================
# Convenience Functions
# ============================================================================


def track_operation(
    operation_type: OperationType, metadata: dict[str, Any] | None = None
):
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


def record_response_time(
    operation_type: OperationType, duration_seconds: float
) -> None:
    """Record response time to global monitor."""
    get_performance_monitor().record_response_time(operation_type, duration_seconds)


def get_performance_report() -> dict[str, Any]:
    """Get performance report from global monitor."""
    return get_performance_monitor().generate_report()


def print_performance_report() -> None:
    """Print performance report."""
    get_performance_monitor().print_report()
