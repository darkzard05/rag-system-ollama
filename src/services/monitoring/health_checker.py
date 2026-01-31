"""
Task 20-2: Health Checking Module
분산 시스템 건강도 체크 및 자동 복구
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Thread
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """노드 건강 상태"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """복구 전략"""

    AUTO_RESTART = "auto_restart"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    NONE = "none"


@dataclass
class HealthCheck:
    """건강 체크"""

    check_id: str
    name: str
    description: str
    check_func: Callable[[], bool]
    interval: float = 10.0  # 초
    timeout: float = 5.0
    critical: bool = False  # True면 실패 시 전체 노드 실패
    enabled: bool = True


@dataclass
class HealthCheckResult:
    """건강 체크 결과"""

    check_id: str
    check_name: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    error_message: str | None = None
    status: HealthStatus = HealthStatus.UNKNOWN


@dataclass
class NodeHealthReport:
    """노드 건강 보고서"""

    node_id: str
    overall_status: HealthStatus
    check_results: list[HealthCheckResult] = field(default_factory=list)
    last_healthy_time: float = field(default_factory=time.time)
    last_unhealthy_time: float | None = None
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    uptime: float = 0.0


class HealthCheckSuite:
    """건강 체크 스위트"""

    def __init__(self):
        self._checks: dict[str, HealthCheck] = {}
        self._results: dict[str, list[HealthCheckResult]] = {}
        self._lock = RLock()

    def register_check(self, health_check: HealthCheck):
        """건강 체크 등록"""
        with self._lock:
            self._checks[health_check.check_id] = health_check
            self._results[health_check.check_id] = []

    def execute_check(self, check_id: str) -> HealthCheckResult | None:
        """건강 체크 실행"""
        with self._lock:
            if check_id not in self._checks:
                return None

            check = self._checks[check_id]

            if not check.enabled:
                return None

        start_time = time.time()
        result = HealthCheckResult(
            check_id=check_id,
            check_name=check.name,
            success=False,
            status=HealthStatus.UNKNOWN,
        )

        try:
            # 타임아웃 설정하여 체크 실행
            success = check.check_func()
            result.success = success
            result.status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.status = HealthStatus.UNHEALTHY

        finally:
            result.duration = time.time() - start_time

            with self._lock:
                if check_id in self._results:
                    self._results[check_id].append(result)
                    # 최근 1000개만 유지
                    if len(self._results[check_id]) > 1000:
                        self._results[check_id] = self._results[check_id][-1000:]

        return result

    def execute_all_checks(self) -> dict[str, HealthCheckResult]:
        """모든 건강 체크 실행"""
        results = {}

        with self._lock:
            check_ids = list(self._checks.keys())

        for check_id in check_ids:
            result = self.execute_check(check_id)
            if result:
                results[check_id] = result

        return results

    def get_check_history(
        self, check_id: str, limit: int = 100
    ) -> list[HealthCheckResult]:
        """체크 이력 조회"""
        with self._lock:
            if check_id in self._results:
                return list(self._results[check_id])[-limit:]
        return []

    def disable_check(self, check_id: str):
        """체크 비활성화"""
        with self._lock:
            if check_id in self._checks:
                self._checks[check_id].enabled = False

    def enable_check(self, check_id: str):
        """체크 활성화"""
        with self._lock:
            if check_id in self._checks:
                self._checks[check_id].enabled = True


class NodeHealthMonitor:
    """노드 건강 모니터"""

    def __init__(self, node_id: str):
        """
        Args:
            node_id: 노드 ID
        """
        self.node_id = node_id
        self._health_suite = HealthCheckSuite()
        self._status = HealthStatus.UNKNOWN
        self._report = NodeHealthReport(
            node_id=node_id, overall_status=HealthStatus.UNKNOWN
        )
        self._lock = RLock()
        self._monitoring_active = False

    def register_health_check(self, health_check: HealthCheck):
        """건강 체크 등록"""
        self._health_suite.register_check(health_check)

    def check_health(self) -> HealthCheckResult:
        """건강 상태 점검"""
        results = self._health_suite.execute_all_checks()

        # 전체 상태 결정
        if not results:
            status = HealthStatus.UNKNOWN
        else:
            critical_failures = sum(1 for r in results.values() if not r.success)

            if critical_failures == len(results):
                status = HealthStatus.UNHEALTHY
            elif critical_failures > 0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

        with self._lock:
            self._status = status
            self._report.overall_status = status
            self._report.check_results = list(results.values())

            if status == HealthStatus.HEALTHY:
                self._report.last_healthy_time = time.time()
                self._report.consecutive_failures = 0
            else:
                self._report.last_unhealthy_time = time.time()
                self._report.consecutive_failures += 1

        # 첫 번째 결과 반환
        return (
            list(results.values())[0]
            if results
            else HealthCheckResult(
                check_id="overall",
                check_name="Overall Health",
                success=status == HealthStatus.HEALTHY,
                status=status,
            )
        )

    def get_status(self) -> HealthStatus:
        """현재 건강 상태"""
        with self._lock:
            return self._status

    def get_report(self) -> NodeHealthReport:
        """건강 보고서"""
        with self._lock:
            # 가동 시간 계산
            if self._report.last_unhealthy_time:
                uptime = (
                    self._report.last_healthy_time - self._report.last_unhealthy_time
                )
            else:
                uptime = time.time() - self._report.last_healthy_time

            self._report.uptime = max(0, uptime)
            return self._report

    def start_monitoring(self, interval: float = 10.0):
        """건강 모니터링 시작"""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                self.check_health()
                time.sleep(interval)

        thread = Thread(target=monitor_loop, daemon=True)
        thread.start()

    def stop_monitoring(self):
        """건강 모니터링 중지"""
        self._monitoring_active = False


class ClusterHealthMonitor:
    """클러스터 건강 모니터"""

    def __init__(self, num_nodes: int = 3):
        """
        Args:
            num_nodes: 노드 개수
        """
        self.num_nodes = num_nodes
        self._node_monitors: dict[str, NodeHealthMonitor] = {
            f"node_{i}": NodeHealthMonitor(f"node_{i}") for i in range(num_nodes)
        }
        self._recovery_strategies: dict[str, RecoveryStrategy] = {
            f"node_{i}": RecoveryStrategy.AUTO_RESTART for i in range(num_nodes)
        }
        self._lock = RLock()

    def check_cluster_health(self) -> dict[str, HealthStatus]:
        """클러스터 전체 건강 상태"""
        statuses = {}

        with self._lock:
            for node_id, monitor in self._node_monitors.items():
                statuses[node_id] = monitor.check_health().status

        return statuses

    def get_cluster_report(self) -> dict[str, Any]:
        """클러스터 보고서"""
        health_statuses = self.check_cluster_health()

        healthy_nodes = sum(
            1 for s in health_statuses.values() if s == HealthStatus.HEALTHY
        )
        degraded_nodes = sum(
            1 for s in health_statuses.values() if s == HealthStatus.DEGRADED
        )
        unhealthy_nodes = sum(
            1 for s in health_statuses.values() if s == HealthStatus.UNHEALTHY
        )

        # 전체 상태 결정
        if unhealthy_nodes > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_nodes > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        node_reports = {}
        with self._lock:
            for node_id, monitor in self._node_monitors.items():
                node_reports[node_id] = {
                    "status": health_statuses[node_id].value,
                    "report": monitor.get_report(),
                }

        return {
            "overall_status": overall_status.value,
            "healthy_nodes": healthy_nodes,
            "degraded_nodes": degraded_nodes,
            "unhealthy_nodes": unhealthy_nodes,
            "node_reports": node_reports,
            "timestamp": time.time(),
        }

    def trigger_recovery(self, node_id: str) -> bool:
        """복구 시작"""
        with self._lock:
            if node_id not in self._node_monitors:
                return False

            strategy = self._recovery_strategies.get(node_id, RecoveryStrategy.NONE)

        if strategy == RecoveryStrategy.AUTO_RESTART:
            return self._restart_node(node_id)
        elif strategy == RecoveryStrategy.GRACEFUL_SHUTDOWN:
            return self._shutdown_node(node_id)
        elif strategy == RecoveryStrategy.FAILOVER:
            return self._failover(node_id)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return self._circuit_break(node_id)

        return False

    def _restart_node(self, node_id: str) -> bool:
        """노드 재시작 시뮬레이션"""
        logger.info(f"Auto-restarting node: {node_id}")
        with self._lock:
            if node_id in self._node_monitors:
                monitor = self._node_monitors[node_id]
                monitor._status = HealthStatus.HEALTHY
                monitor._report.overall_status = HealthStatus.HEALTHY
                monitor._report.last_healthy_time = time.time()
                monitor._report.recovery_attempts += 1
                return True
        return False

    def _shutdown_node(self, node_id: str) -> bool:
        """노드 종료 시뮬레이션"""
        logger.warning(f"Gracefully shutting down node: {node_id}")
        with self._lock:
            if node_id in self._node_monitors:
                self._node_monitors[node_id]._status = HealthStatus.UNKNOWN
                return True
        return False

    def _failover(self, node_id: str) -> bool:
        """페일오버 시뮬레이션"""
        logger.warning(f"Executing failover for node: {node_id}")
        return True

    def _circuit_break(self, node_id: str) -> bool:
        """서킷 브레이커 활성화 시뮬레이션"""
        logger.warning(f"Activating circuit breaker for node: {node_id}")
        return True

    def set_recovery_strategy(self, node_id: str, strategy: RecoveryStrategy):
        """복구 전략 설정"""
        with self._lock:
            self._recovery_strategies[node_id] = strategy

    def start_monitoring_all(self, interval: float = 10.0):
        """모든 노드 모니터링 시작"""
        with self._lock:
            for monitor in self._node_monitors.values():
                monitor.start_monitoring(interval)

    def stop_monitoring_all(self):
        """모든 노드 모니터링 중지"""
        with self._lock:
            for monitor in self._node_monitors.values():
                monitor.stop_monitoring()


class HealthCheckFactory:
    """건강 체크 팩토리"""

    @staticmethod
    def create_cpu_check(threshold: float = 80.0) -> HealthCheck:
        """CPU 체크"""

        def cpu_check() -> bool:
            # 모의 CPU 사용률
            cpu_usage = random.uniform(10, 90)
            return cpu_usage < threshold

        return HealthCheck(
            check_id="cpu_check",
            name="CPU Usage Check",
            description=f"CPU 사용률이 {threshold}% 이하인지 확인",
            check_func=cpu_check,
            critical=True,
        )

    @staticmethod
    def create_memory_check(threshold: float = 85.0) -> HealthCheck:
        """메모리 체크"""

        def memory_check() -> bool:
            # 모의 메모리 사용률
            memory_usage = random.uniform(30, 95)
            return memory_usage < threshold

        return HealthCheck(
            check_id="memory_check",
            name="Memory Usage Check",
            description=f"메모리 사용률이 {threshold}% 이하인지 확인",
            check_func=memory_check,
            critical=True,
        )

    @staticmethod
    def create_disk_check(threshold: float = 90.0) -> HealthCheck:
        """디스크 체크"""

        def disk_check() -> bool:
            # 모의 디스크 사용률
            disk_usage = random.uniform(20, 85)
            return disk_usage < threshold

        return HealthCheck(
            check_id="disk_check",
            name="Disk Usage Check",
            description=f"디스크 사용률이 {threshold}% 이하인지 확인",
            check_func=disk_check,
            critical=False,
        )

    @staticmethod
    def create_service_check() -> HealthCheck:
        """서비스 가용성 체크"""

        def service_check() -> bool:
            # 모의 서비스 상태
            return random.random() > 0.1  # 90% 성공률

        return HealthCheck(
            check_id="service_check",
            name="Service Availability Check",
            description="서비스가 정상적으로 작동하는지 확인",
            check_func=service_check,
            critical=True,
        )

    @staticmethod
    def create_connectivity_check() -> HealthCheck:
        """연결성 체크"""

        def connectivity_check() -> bool:
            # 모의 네트워크 연결성
            return random.random() > 0.05  # 95% 성공률

        return HealthCheck(
            check_id="connectivity_check",
            name="Connectivity Check",
            description="네트워크 연결이 정상인지 확인",
            check_func=connectivity_check,
            critical=True,
        )
