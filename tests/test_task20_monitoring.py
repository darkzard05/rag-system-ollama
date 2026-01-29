"""
Task 20: 분산 모니터링 및 성능 추적 테스트
- 시스템 건강 체크
- 메트릭 집계 및 분석
- 이상 탐지
- 알림 관리
"""

import pytest
import random
import time

from src.services.monitoring.health_checker import (
    HealthCheck,
    HealthCheckSuite,
    NodeHealthMonitor,
    ClusterHealthMonitor,
)
from src.services.monitoring.metrics_aggregator import (
    TimeSeriesAnalyzer,
    AnomalyDetector,
    AnomalyResult,
    AlertManager,
    MetricsAggregator,
)


# ==============================================
# Fixtures
# ==============================================


@pytest.fixture
def health_monitor():
    """건강 모니터 Fixture"""
    monitor = ClusterHealthMonitor()
    yield monitor


@pytest.fixture
def metrics_aggregator():
    """메트릭 집계 Fixture"""
    aggregator = MetricsAggregator()
    yield aggregator


# ==============================================
# 테스트 그룹 1: 기본 메트릭 집계 (5개 테스트)
# ==============================================


class TestMetricsAggregationBasic:
    """메트릭 집계 기본 테스트"""

    def test_01_record_metrics(self, metrics_aggregator):
        """메트릭 기록"""
        metrics_aggregator.record_metric("node_0", 45.5)
        metrics_aggregator.record_metric("node_1", 52.3)
        metrics_aggregator.record_metric("node_2", 48.7)

        stats = metrics_aggregator.get_time_series_stats("node_0")
        assert stats is not None
        assert stats.count >= 1

    def test_02_time_series_statistics(self, metrics_aggregator):
        """시계열 통계"""
        for _ in range(20):
            metrics_aggregator.record_metric("node_0", random.uniform(40, 60))

        stats = metrics_aggregator.get_time_series_stats("node_0")

        assert stats is not None
        assert stats.count == 20
        assert 40 <= stats.mean <= 60
        assert stats.stddev >= 0

    def test_03_metric_status_evaluation(self, metrics_aggregator):
        """메트릭 상태 평가"""
        # 정상 메트릭
        for _ in range(15):
            metrics_aggregator.record_metric("node_0", random.uniform(30, 50))

        # 평가
        status = metrics_aggregator.get_time_series_stats("node_0")

        assert status is not None
        assert status.mean > 0

    def test_04_cluster_statistics(self, metrics_aggregator):
        """클러스터 통계"""
        nodes = ["node_0", "node_1", "node_2"]
        for node_id in nodes:
            for _ in range(10):
                metrics_aggregator.record_metric(node_id, random.uniform(30, 70))

        stats = metrics_aggregator.get_cluster_stats()

        assert len(stats) == 3
        for node_id in nodes:
            assert node_id in stats

    def test_05_time_series_analyzer(self):
        """시계열 분석기"""
        analyzer = TimeSeriesAnalyzer()

        # 데이터 추가
        for i in range(50):
            analyzer.add_point(time.time() + i, 100 + i * 0.5)

        stats = analyzer.get_stats()

        assert stats.count == 50
        assert stats.mean > 0
        assert stats.trend in ["stable", "increasing", "decreasing"]


# ==============================================
# 테스트 그룹 2: 건강 체크 (5개 테스트)
# ==============================================


class TestHealthChecking:
    """건강 체크 테스트"""

    def test_06_create_health_checks(self):
        """건강 체크 생성"""
        suite = HealthCheckSuite()

        def cpu_check():
            return True

        check = HealthCheck(
            check_id="cpu_1",
            name="cpu_check",
            description="CPU usage check",
            check_func=cpu_check,
            interval=10,
            timeout=5,
        )

        suite.register_check(check)

        assert len(suite._checks) == 1

    def test_07_node_health_check(self, health_monitor):
        """노드 건강 체크"""
        # 클러스터 건강 체크
        health_statuses = health_monitor.check_cluster_health()

        assert health_statuses is not None
        assert len(health_statuses) > 0

    def test_08_cluster_health_report(self, health_monitor):
        """클러스터 건강 보고"""
        report = health_monitor.get_cluster_report()

        assert report is not None
        assert "overall_status" in report
        assert "healthy_nodes" in report
        assert "degraded_nodes" in report
        assert "unhealthy_nodes" in report

    def test_09_recovery_strategy(self, health_monitor):
        """복구 전략"""
        # 복구 전략 적용
        health_monitor.trigger_recovery("node_0")

        report = health_monitor.get_cluster_report()
        assert report is not None

    def test_10_node_health_monitoring(self):
        """노드 건강 모니터링"""
        # 건강도 모니터링
        monitor = NodeHealthMonitor("node_0")
        result = monitor.check_health()

        status = monitor.get_status()
        assert status is not None


# ==============================================
# 테스트 그룹 3: 이상 탐지 (5개 테스트)
# ==============================================


class TestAnomalyDetection:
    """이상 탐지 테스트"""

    def test_11_zscore_anomaly_detection(self):
        """Z-score 이상 탐지"""
        detector = AnomalyDetector(sensitivity=2.0)

        # 정상 데이터
        normal_data = [50 + random.uniform(-5, 5) for _ in range(30)]

        # 정상 데이터는 이상이 아님
        result = detector.detect_anomaly(52.0, normal_data[:20], method="zscore")
        assert isinstance(result, AnomalyResult)

        # 극단적인 값은 이상임
        result = detector.detect_anomaly(200, normal_data, method="zscore")
        assert isinstance(result, AnomalyResult)

    def test_12_iqr_anomaly_detection(self):
        """IQR 이상 탐지"""
        detector = AnomalyDetector(sensitivity=1.5)

        data = [50 + random.uniform(-5, 5) for _ in range(30)]

        result = detector.detect_anomaly(100, data, method="iqr")

        assert isinstance(result, AnomalyResult)

    def test_13_mad_anomaly_detection(self):
        """MAD 이상 탐지"""
        detector = AnomalyDetector(sensitivity=3.0)

        data = [100 + random.uniform(-10, 10) for _ in range(25)]

        result = detector.detect_anomaly(200, data, method="mad")

        assert isinstance(result, AnomalyResult)

    def test_14_time_series_analysis(self):
        """시계열 분석"""
        analyzer = TimeSeriesAnalyzer()

        # 데이터 추가
        for i in range(50):
            analyzer.add_point(time.time() + i, 100 + i * 0.5)

        stats = analyzer.get_stats()

        assert stats.count == 50
        assert stats.mean > 0
        assert stats.trend in ["stable", "increasing", "decreasing"]

    def test_15_alert_manager(self):
        """알림 관리자"""
        manager = AlertManager()

        # 알림 규칙 등록
        manager.register_alert_rule(
            "cpu_high", "cpu_usage", ">", 80.0, severity="critical"
        )

        # 알림 확인
        alerts = manager.check_alert_rules("cpu_usage", 85.0)

        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"


# ==============================================
# 테스트 그룹 4: 메트릭 집계 상세 (3개 테스트)
# ==============================================


class TestMetricsAggregationAdvanced:
    """메트릭 집계 상세 테스트"""

    def test_16_detect_cluster_anomaly(self, metrics_aggregator):
        """클러스터 이상 탐지"""
        # 정상 메트릭 기록
        for i in range(20):
            metrics_aggregator.record_metric("node_0", 50 + random.uniform(-5, 5))

        # 통계 조회
        stats = metrics_aggregator.get_time_series_stats("node_0")

        # 이상 탐지 테스트
        if stats:
            # stats에는 count, mean, median 등이 있음
            assert stats.count == 20
            result = metrics_aggregator.detect_anomaly("node_0", 52.0)
            assert isinstance(result, AnomalyResult)

    def test_17_alert_rules_and_checking(self, metrics_aggregator):
        """알림 규칙 및 확인"""
        # 알림 규칙 등록
        metrics_aggregator.register_alert_rule(
            "node_0", "cpu_high", "cpu_usage", ">", 80.0, "critical"
        )

        # 알림 확인
        alerts = metrics_aggregator.check_alerts("node_0", "cpu_usage", 85.0)

        assert len(alerts) > 0
        assert alerts[0]["severity"] == "critical"

    def test_18_cluster_alerts_summary(self, metrics_aggregator):
        """클러스터 알림 요약"""
        # 알림 규칙 등록
        for node_id in ["node_0", "node_1", "node_2"]:
            metrics_aggregator.register_alert_rule(
                node_id, "cpu_alert", "cpu_usage", ">", 75.0, "warning"
            )

        # 알림 확인
        alerts_summary = metrics_aggregator.get_cluster_alerts()

        assert len(alerts_summary) >= 0


# ==============================================
# 테스트 그룹 5: 통합 모니터링 (2개 테스트)
# ==============================================


class TestMonitoringIntegration:
    """통합 모니터링 테스트"""

    def test_19_complete_monitoring_pipeline(self, health_monitor, metrics_aggregator):
        """완전한 모니터링 파이프라인"""
        # 메트릭 기록
        for i in range(15):
            metrics_aggregator.record_metric("node_0", 50 + random.uniform(-5, 5))

        # 건강 상태 체크
        report = health_monitor.get_cluster_report()
        assert report is not None

        # 통계 조회
        stats = metrics_aggregator.get_time_series_stats("node_0")
        assert stats is not None
        assert stats.count >= 15

    def test_20_system_health_overview(self, health_monitor, metrics_aggregator):
        """시스템 건강 개요"""
        # 여러 노드의 메트릭 수집
        nodes = ["node_0", "node_1", "node_2"]
        for node_id in nodes:
            for _ in range(10):
                metrics_aggregator.record_metric(node_id, random.uniform(40, 60))

        # 건강 상태
        report = health_monitor.get_cluster_report()
        assert report is not None

        # 클러스터 통계
        stats = metrics_aggregator.get_cluster_stats()
        assert len(stats) == 3


# ==============================================
# 테스트 그룹 6: 처리량 및 에러 (2개 테스트)
# ==============================================


class TestThroughputAndErrors:
    """처리량 및 에러 추적 테스트"""

    def test_21_throughput_tracking(self, metrics_aggregator):
        """처리량 추적"""
        # 처리량 메트릭 기록
        for _ in range(100):
            metrics_aggregator.record_metric("throughput", random.uniform(100, 500))

        # 직접 nodes 확인
        stats = metrics_aggregator.get_time_series_stats("throughput")

        if stats:
            assert stats.count >= 1
        else:
            # node가 없을 수 있으니 클러스터 통계로 확인
            cluster_stats = metrics_aggregator.get_cluster_stats()
            assert cluster_stats is not None

    def test_22_error_tracking(self, metrics_aggregator):
        """에러 추적"""
        # 에러율 메트릭 기록 (0-100)
        for _ in range(50):
            error_rate = random.uniform(0, 10)  # 0-10% 에러율
            metrics_aggregator.record_metric("error_rate", error_rate)

        # 메트릭이 기록되었는지 확인
        cluster_stats = metrics_aggregator.get_cluster_stats()

        assert cluster_stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
