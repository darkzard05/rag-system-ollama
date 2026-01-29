"""
Task 20-3: Metrics Aggregation Module
메트릭 집계 및 분석 - 시계열 분석, 이상 탐지, 알림
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import time
from threading import RLock
from collections import deque
import statistics


class AnomalyType(Enum):
    """이상 타입"""

    SPIKE = "spike"  # 급증
    DIP = "dip"  # 급락
    TREND = "trend"  # 추세 변화
    OUTLIER = "outlier"  # 아웃라이어
    SEASONALITY = "seasonality"  # 계절성 이상


@dataclass
class TimeSeriesPoint:
    """시계열 데이터 포인트"""

    timestamp: float
    value: float
    label: str = ""


@dataclass
class AnomalyResult:
    """이상 탐지 결과"""

    is_anomaly: bool
    anomaly_type: Optional[AnomalyType] = None
    confidence: float = 0.0
    expected_value: float = 0.0
    deviation: float = 0.0
    message: str = ""


@dataclass
class TimeSeriesStats:
    """시계열 통계"""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    trend: str = "stable"  # stable, increasing, decreasing
    volatility: float = 0.0


class TimeSeriesAnalyzer:
    """시계열 분석기"""

    def __init__(self, max_points: int = 1000):
        """
        Args:
            max_points: 최대 데이터 포인트
        """
        self.max_points = max_points
        self._data: deque = deque(maxlen=max_points)
        self._lock = RLock()

    def add_point(self, timestamp: float, value: float, label: str = ""):
        """데이터 포인트 추가"""
        point = TimeSeriesPoint(timestamp=timestamp, value=value, label=label)

        with self._lock:
            self._data.append(point)

    def get_stats(self) -> TimeSeriesStats:
        """시계열 통계"""
        with self._lock:
            if not self._data:
                return TimeSeriesStats()

            values = [p.value for p in self._data]

        stats = TimeSeriesStats(count=len(values))
        stats.mean = statistics.mean(values)
        stats.median = statistics.median(values)
        stats.min = min(values)
        stats.max = max(values)

        if len(values) > 1:
            stats.stddev = statistics.stdev(values)

        sorted_values = sorted(values)
        q25_idx = len(sorted_values) // 4
        q75_idx = (3 * len(sorted_values)) // 4
        stats.p25 = sorted_values[q25_idx]
        stats.p75 = sorted_values[q75_idx]

        # 추세 계산
        stats.trend = self._calculate_trend(values)

        # 변동성 계산
        stats.volatility = self._calculate_volatility(values)

        return stats

    def _calculate_trend(self, values: List[float]) -> str:
        """추세 계산"""
        if len(values) < 2:
            return "stable"

        # 선형 회귀 기울기
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def _calculate_volatility(self, values: List[float]) -> float:
        """변동성 계산"""
        if len(values) < 2:
            return 0.0

        returns = [
            (values[i] - values[i - 1]) / values[i - 1] if values[i - 1] != 0 else 0
            for i in range(1, len(values))
        ]

        if not returns:
            return 0.0

        return statistics.stdev(returns) if len(returns) > 1 else 0.0

    def get_data(self, limit: Optional[int] = None) -> List[TimeSeriesPoint]:
        """데이터 조회"""
        with self._lock:
            data = list(self._data)

        if limit:
            data = data[-limit:]

        return data


class AnomalyDetector:
    """이상 탐지기"""

    def __init__(self, sensitivity: float = 2.0):
        """
        Args:
            sensitivity: 감도 (높을수록 더 엄격)
        """
        self.sensitivity = sensitivity
        self._lock = RLock()

    def detect_anomaly(
        self,
        current_value: float,
        historical_values: List[float],
        method: str = "zscore",
    ) -> AnomalyResult:
        """이상 탐지"""
        if not historical_values:
            return AnomalyResult(is_anomaly=False)

        if method == "zscore":
            return self._detect_zscore(current_value, historical_values)
        elif method == "iqr":
            return self._detect_iqr(current_value, historical_values)
        elif method == "mad":
            return self._detect_mad(current_value, historical_values)
        else:
            return AnomalyResult(is_anomaly=False)

    def _detect_zscore(
        self, current_value: float, historical_values: List[float]
    ) -> AnomalyResult:
        """Z-score 방법"""
        if len(historical_values) < 2:
            return AnomalyResult(is_anomaly=False)

        mean = statistics.mean(historical_values)
        stddev = statistics.stdev(historical_values)

        if stddev == 0:
            return AnomalyResult(is_anomaly=False)

        z_score = abs((current_value - mean) / stddev)
        threshold = self.sensitivity

        is_anomaly = z_score > threshold
        confidence = min(1.0, z_score / (threshold * 2))
        deviation = current_value - mean

        anomaly_type = None
        if is_anomaly:
            if current_value > mean:
                anomaly_type = AnomalyType.SPIKE
            else:
                anomaly_type = AnomalyType.DIP

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence,
            expected_value=mean,
            deviation=deviation,
            message=f"Z-score: {z_score:.2f}, threshold: {threshold}",
        )

    def _detect_iqr(
        self, current_value: float, historical_values: List[float]
    ) -> AnomalyResult:
        """Interquartile Range 방법"""
        if len(historical_values) < 4:
            return AnomalyResult(is_anomaly=False)

        sorted_values = sorted(historical_values)
        q1 = sorted_values[len(sorted_values) // 4]
        q3 = sorted_values[(3 * len(sorted_values)) // 4]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_anomaly = current_value < lower_bound or current_value > upper_bound

        anomaly_type = None
        if is_anomaly:
            if current_value > upper_bound:
                anomaly_type = AnomalyType.SPIKE
            else:
                anomaly_type = AnomalyType.DIP

        confidence = 0.8 if is_anomaly else 0.0
        mean = statistics.mean(historical_values)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence,
            expected_value=mean,
            deviation=current_value - mean,
            message=f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]",
        )

    def _detect_mad(
        self, current_value: float, historical_values: List[float]
    ) -> AnomalyResult:
        """Median Absolute Deviation 방법"""
        if not historical_values:
            return AnomalyResult(is_anomaly=False)

        median = statistics.median(historical_values)
        deviations = [abs(v - median) for v in historical_values]
        mad = statistics.median(deviations)

        if mad == 0:
            return AnomalyResult(is_anomaly=False)

        modified_z_score = 0.6745 * (current_value - median) / mad
        threshold = self.sensitivity

        is_anomaly = abs(modified_z_score) > threshold
        confidence = min(1.0, abs(modified_z_score) / (threshold * 2))

        anomaly_type = None
        if is_anomaly:
            if current_value > median:
                anomaly_type = AnomalyType.SPIKE
            else:
                anomaly_type = AnomalyType.DIP

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence,
            expected_value=median,
            deviation=current_value - median,
            message=f"Modified Z-score: {modified_z_score:.2f}",
        )


class AlertManager:
    """알림 관리자"""

    def __init__(self):
        self._alerts: deque = deque(maxlen=1000)
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def register_alert_rule(
        self,
        rule_id: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
    ):
        """알림 규칙 등록"""
        with self._lock:
            self._alert_rules[rule_id] = {
                "metric_name": metric_name,
                "condition": condition,  # >, <, ==, !=, >=, <=
                "threshold": threshold,
                "severity": severity,
                "triggered_count": 0,
                "last_triggered": None,
            }

    def check_alert_rules(
        self, metric_name: str, current_value: float
    ) -> List[Dict[str, Any]]:
        """알림 규칙 확인"""
        triggered_alerts = []

        with self._lock:
            for rule_id, rule in self._alert_rules.items():
                if rule["metric_name"] != metric_name:
                    continue

                if self._evaluate_condition(
                    current_value, rule["condition"], rule["threshold"]
                ):
                    alert = {
                        "rule_id": rule_id,
                        "metric_name": metric_name,
                        "current_value": current_value,
                        "threshold": rule["threshold"],
                        "condition": rule["condition"],
                        "severity": rule["severity"],
                        "timestamp": time.time(),
                    }

                    triggered_alerts.append(alert)
                    self._alerts.append(alert)

                    rule["triggered_count"] += 1
                    rule["last_triggered"] = time.time()

        return triggered_alerts

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """조건 평가"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.0001
        elif condition == "!=":
            return abs(value - threshold) >= 0.0001
        else:
            return False

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """알림 조회"""
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """알림 요약"""
        with self._lock:
            alerts = list(self._alerts)

        if not alerts:
            return {"total_alerts": 0, "critical": 0, "warning": 0, "info": 0}

        severity_counts = {}
        for alert in alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_alerts": len(alerts),
            "critical": severity_counts.get("critical", 0),
            "warning": severity_counts.get("warning", 0),
            "info": severity_counts.get("info", 0),
            "severity_distribution": severity_counts,
        }


class MetricsAggregator:
    """메트릭 집계기"""

    def __init__(self, num_nodes: int = 3):
        """
        Args:
            num_nodes: 노드 개수
        """
        self.num_nodes = num_nodes
        self._time_series_analyzers: Dict[str, TimeSeriesAnalyzer] = {
            f"node_{i}": TimeSeriesAnalyzer() for i in range(num_nodes)
        }
        self._anomaly_detectors: Dict[str, AnomalyDetector] = {
            f"node_{i}": AnomalyDetector() for i in range(num_nodes)
        }
        self._alert_managers: Dict[str, AlertManager] = {
            f"node_{i}": AlertManager() for i in range(num_nodes)
        }
        self._lock = RLock()

    def record_metric(self, node_id: str, value: float, label: str = ""):
        """메트릭 기록"""
        if node_id in self._time_series_analyzers:
            self._time_series_analyzers[node_id].add_point(time.time(), value, label)

    def get_time_series_stats(self, node_id: str) -> Optional[TimeSeriesStats]:
        """시계열 통계"""
        if node_id in self._time_series_analyzers:
            return self._time_series_analyzers[node_id].get_stats()
        return None

    def detect_anomaly(
        self, node_id: str, current_value: float, method: str = "zscore"
    ) -> AnomalyResult:
        """이상 탐지"""
        if node_id not in self._time_series_analyzers:
            return AnomalyResult(is_anomaly=False)

        analyzer = self._time_series_analyzers[node_id]
        detector = self._anomaly_detectors[node_id]

        data_points = analyzer.get_data()
        historical_values = [p.value for p in data_points]

        return detector.detect_anomaly(current_value, historical_values, method)

    def register_alert_rule(
        self,
        node_id: str,
        rule_id: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
    ):
        """알림 규칙 등록"""
        if node_id in self._alert_managers:
            self._alert_managers[node_id].register_alert_rule(
                rule_id, metric_name, condition, threshold, severity
            )

    def check_alerts(
        self, node_id: str, metric_name: str, current_value: float
    ) -> List[Dict]:
        """알림 확인"""
        if node_id in self._alert_managers:
            return self._alert_managers[node_id].check_alert_rules(
                metric_name, current_value
            )
        return []

    def get_cluster_stats(self) -> Dict[str, Any]:
        """클러스터 통계"""
        stats = {}

        with self._lock:
            for node_id, analyzer in self._time_series_analyzers.items():
                node_stats = analyzer.get_stats()
                stats[node_id] = {
                    "mean": node_stats.mean,
                    "median": node_stats.median,
                    "stddev": node_stats.stddev,
                    "min": node_stats.min,
                    "max": node_stats.max,
                    "trend": node_stats.trend,
                    "volatility": node_stats.volatility,
                }

        return stats

    def get_cluster_alerts(self) -> Dict[str, Dict]:
        """클러스터 알림"""
        alerts = {}

        with self._lock:
            for node_id, alert_manager in self._alert_managers.items():
                alerts[node_id] = alert_manager.get_alert_summary()

        return alerts
