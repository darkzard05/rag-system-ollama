"""
Advanced Performance Monitoring System for RAG
"""

import logging
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class BottleneckType(Enum):
    """Bottleneck Detection Types"""

    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"


class HealthStatus(Enum):
    """System Health Status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Advanced performance metric"""

    name: str
    value: float
    unit: str
    timestamp: float
    component: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BottleneckReport:
    """Bottleneck analysis report"""

    bottleneck_id: str
    bottleneck_type: BottleneckType
    component: str
    severity: str  # low, medium, high, critical
    metric_name: str
    current_value: float
    threshold: float
    recommendation: str
    detected_at: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["bottleneck_type"] = self.bottleneck_type.value
        return data


class AdvancedPerformanceMonitor:
    """Advanced performance monitoring with bottleneck detection"""

    def __init__(self):
        """Initialize advanced monitor"""
        self.metrics: list[PerformanceMetric] = []
        self.bottleneck_reports: list[BottleneckReport] = []
        self.component_metrics: dict[str, list[float]] = defaultdict(list)
        self.health_history: list[tuple] = []

        # Thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "query_latency": 500.0,
            "cache_miss_rate": 0.5,
            "error_rate": 0.01,
        }

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        component: str = "",
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Record performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=time.time(),
                component=component,
                context=context or {},
            )

            self.metrics.append(metric)

            if component:
                self.component_metrics[component].append(value)

            # Check for bottleneck
            if name in self.thresholds:
                if value > self.thresholds[name]:
                    self._detect_bottleneck(name, value, component)

            return True

    def _detect_bottleneck(self, metric_name: str, value: float, component: str):
        """Detect and report bottleneck"""
        threshold = self.thresholds[metric_name]

        # Determine severity
        severity = "medium"
        if value > threshold * 2.0:
            severity = "critical"
        elif value > threshold * 1.5:
            severity = "high"

        # Determine bottleneck type
        bottleneck_type = BottleneckType.CPU_INTENSIVE
        recommendation = "Optimize query patterns"

        if "memory" in metric_name:
            bottleneck_type = BottleneckType.MEMORY_INTENSIVE
            recommendation = "Consider adding more memory or clearing caches"
        elif "latency" in metric_name:
            bottleneck_type = BottleneckType.IO_BOUND
            recommendation = "Add indexes or optimize database queries"
        elif "error" in metric_name:
            recommendation = "Check system logs for error patterns"

        report = BottleneckReport(
            bottleneck_id=f"bnk_{len(self.bottleneck_reports)}",
            bottleneck_type=bottleneck_type,
            component=component,
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold=threshold,
            recommendation=recommendation,
            detected_at=time.time(),
        )

        self.bottleneck_reports.append(report)
        self.logger.warning(
            f"Bottleneck detected: {metric_name} = {value} (threshold: {threshold})"
        )

    def get_component_summary(self, component: str) -> dict[str, Any]:
        """Get component performance summary"""
        with self._lock:
            if component not in self.component_metrics:
                return {}

            values = self.component_metrics[component]
            if not values:
                return {}

            values.sort()
            return {
                "component": component,
                "sample_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "p95": values[int(len(values) * 0.95)],
                "p99": values[int(len(values) * 0.99)],
            }

    def get_bottleneck_report(self) -> list[dict[str, Any]]:
        """Get bottleneck report"""
        with self._lock:
            return [b.to_dict() for b in self.bottleneck_reports[-50:]]

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health assessment"""
        with self._lock:
            critical_count = sum(
                1 for b in self.bottleneck_reports if b.severity == "critical"
            )
            high_count = sum(1 for b in self.bottleneck_reports if b.severity == "high")

            if critical_count > 0:
                status = HealthStatus.CRITICAL.value
            elif high_count > 2:
                status = HealthStatus.DEGRADED.value
            else:
                status = HealthStatus.HEALTHY.value

            return {
                "status": status,
                "critical_issues": critical_count,
                "high_issues": high_count,
                "total_bottlenecks": len(self.bottleneck_reports),
                "metrics_collected": len(self.metrics),
            }

    def get_performance_trend(
        self, metric_name: str, minutes: int = 60
    ) -> dict[str, Any]:
        """Analyze performance trend"""
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)

            recent_metrics = [
                m
                for m in self.metrics
                if m.name == metric_name and m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {}

            values = [m.value for m in recent_metrics]
            values.sort()

            # Determine trend
            if len(values) >= 2:
                first_half = statistics.mean(values[: len(values) // 2])
                second_half = statistics.mean(values[len(values) // 2 :])
                trend = "improving" if second_half < first_half else "degrading"
            else:
                trend = "unknown"

            return {
                "metric_name": metric_name,
                "period_minutes": minutes,
                "sample_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "trend": trend,
            }
