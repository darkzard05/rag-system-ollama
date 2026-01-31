"""
클러스터 관리 - 노드 모니터링 및 리소스 추적.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Thread

import psutil

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """노드 상태."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class NodeMetrics:
    """노드 메트릭."""

    node_id: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    cpu_count: int = 0
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    def update(self):
        """메트릭 업데이트."""
        self.last_heartbeat = time.time()

        try:
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            self.memory_percent = psutil.virtual_memory().percent
            self.memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            self.cpu_count = psutil.cpu_count()
        except Exception as e:
            logger.error(f"메트릭 수집 실패: {e}")

    def is_healthy(
        self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0
    ) -> bool:
        """노드 상태 확인."""
        return (
            self.cpu_percent < cpu_threshold and self.memory_percent < memory_threshold
        )


@dataclass
class HealthCheckConfig:
    """헬스 체크 설정."""

    interval_seconds: float = 5.0
    heartbeat_timeout: float = 30.0
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    enabled: bool = True


class ClusterManager:
    """클러스터 관리자."""

    def __init__(self, num_nodes: int = 1, config: HealthCheckConfig | None = None):
        """초기화.

        Args:
            num_nodes: 노드 수
            config: 헬스 체크 설정
        """
        self.num_nodes = num_nodes
        self.config = config or HealthCheckConfig()

        self._lock = RLock()
        self._nodes: dict[str, NodeMetrics] = {}
        self._health_monitor_thread: Thread | None = None
        self._stop_monitoring = False

        # 로컬 머신을 노드로 등록
        self._init_nodes()

        # 헬스 체크 시작
        if self.config.enabled:
            self.start_health_monitoring()

    def _init_nodes(self):
        """노드 초기화."""
        with self._lock:
            for i in range(self.num_nodes):
                node_id = f"node-{i}"
                node = NodeMetrics(node_id=node_id)
                node.update()
                self._nodes[node_id] = node

            logger.info(f"클러스터 초기화: {self.num_nodes}개 노드")

    def start_health_monitoring(self):
        """헬스 체크 시작."""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            logger.warning("헬스 모니터링이 이미 실행 중입니다")
            return

        self._stop_monitoring = False
        self._health_monitor_thread = Thread(
            target=self._monitor_health,
            daemon=True,
        )
        self._health_monitor_thread.start()
        logger.info("헬스 모니터링 시작")

    def stop_health_monitoring(self):
        """헬스 체크 중지."""
        self._stop_monitoring = True

        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5.0)

        logger.info("헬스 모니터링 중지")

    def _monitor_health(self):
        """헬스 모니터링 루프."""
        while not self._stop_monitoring:
            try:
                time.sleep(self.config.interval_seconds)

                with self._lock:
                    for _node_id, node in self._nodes.items():
                        # 메트릭 업데이트
                        node.update()

                        # 상태 판단
                        if not node.is_healthy(
                            self.config.cpu_threshold,
                            self.config.memory_threshold,
                        ):
                            node.status = NodeStatus.DEGRADED
                        else:
                            node.status = NodeStatus.HEALTHY

                        # 타임아웃 확인
                        time_since_heartbeat = time.time() - node.last_heartbeat
                        if time_since_heartbeat > self.config.heartbeat_timeout:
                            node.status = NodeStatus.OFFLINE

            except Exception as e:
                logger.error(f"헬스 모니터링 오류: {e}")

    def get_node_metrics(self, node_id: str) -> NodeMetrics | None:
        """노드 메트릭 조회."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes_metrics(self) -> dict[str, NodeMetrics]:
        """모든 노드 메트릭."""
        with self._lock:
            return dict(self._nodes)

    def get_healthy_nodes(self) -> list[str]:
        """정상 노드 목록."""
        with self._lock:
            return [
                node_id
                for node_id, node in self._nodes.items()
                if node.status == NodeStatus.HEALTHY
            ]

    def get_cluster_status(self) -> dict:
        """클러스터 상태."""
        with self._lock:
            nodes = self._nodes

            avg_cpu = (
                sum(n.cpu_percent for n in nodes.values()) / len(nodes) if nodes else 0
            )
            avg_memory = (
                sum(n.memory_percent for n in nodes.values()) / len(nodes)
                if nodes
                else 0
            )

            healthy_count = sum(
                1 for n in nodes.values() if n.status == NodeStatus.HEALTHY
            )
            degraded_count = sum(
                1 for n in nodes.values() if n.status == NodeStatus.DEGRADED
            )
            offline_count = sum(
                1 for n in nodes.values() if n.status == NodeStatus.OFFLINE
            )

            return {
                "total_nodes": len(nodes),
                "healthy_nodes": healthy_count,
                "degraded_nodes": degraded_count,
                "offline_nodes": offline_count,
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
            }

    def update_node_job_status(
        self,
        node_id: str,
        active: int = 0,
        completed: int = 0,
        failed: int = 0,
    ):
        """노드의 작업 상태 업데이트."""
        with self._lock:
            if node_id not in self._nodes:
                return

            node = self._nodes[node_id]
            node.active_jobs += active
            node.completed_jobs += completed
            node.failed_jobs += failed

    def get_node_load(self, node_id: str) -> float | None:
        """노드 부하 (0.0 - 1.0)."""
        with self._lock:
            if node_id not in self._nodes:
                return None

            node = self._nodes[node_id]

            # CPU 사용률 + 메모리 사용률의 평균
            load = (node.cpu_percent + node.memory_percent) / 200.0
            return min(load, 1.0)

    def select_best_node(self) -> str | None:
        """가장 부하가 낮은 노드 선택."""
        with self._lock:
            healthy_nodes = {
                node_id: node
                for node_id, node in self._nodes.items()
                if node.status == NodeStatus.HEALTHY
            }

            if not healthy_nodes:
                return None

            # 가장 부하가 낮은 노드 선택
            best_node = min(
                healthy_nodes.items(),
                key=lambda x: (x[1].cpu_percent + x[1].memory_percent) / 2,
            )

            return best_node[0]

    def get_nodes_sorted_by_load(self) -> list[str]:
        """노드를 부하 순서로 정렬."""
        with self._lock:
            sorted_nodes = sorted(
                self._nodes.items(),
                key=lambda x: (x[1].cpu_percent + x[1].memory_percent) / 2,
            )

            return [node_id for node_id, _ in sorted_nodes]

    def shutdown(self):
        """클러스터 종료."""
        self.stop_health_monitoring()
        logger.info("클러스터 종료")


class NodeMonitor:
    """개별 노드 모니터."""

    def __init__(self, node_id: str, check_interval: float = 5.0):
        self.node_id = node_id
        self.check_interval = check_interval
        self._metrics = NodeMetrics(node_id=node_id)
        self._lock = RLock()

    def collect_metrics(self) -> NodeMetrics:
        """메트릭 수집."""
        with self._lock:
            self._metrics.update()
            return self._metrics

    def get_summary(self) -> dict:
        """요약 정보."""
        with self._lock:
            return {
                "node_id": self.node_id,
                "cpu_percent": self._metrics.cpu_percent,
                "memory_percent": self._metrics.memory_percent,
                "memory_mb": self._metrics.memory_mb,
                "status": self._metrics.status.value,
            }


class ResourceTracker:
    """리소스 추적자."""

    def __init__(self):
        self._lock = RLock()
        self._resource_history: dict[str, list[dict]] = {}
        self._max_history_size = 1000

    def record_resource_usage(self, node_id: str, metrics: NodeMetrics):
        """리소스 사용량 기록."""
        with self._lock:
            if node_id not in self._resource_history:
                self._resource_history[node_id] = []

            record = {
                "timestamp": time.time(),
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_mb": metrics.memory_mb,
                "active_jobs": metrics.active_jobs,
            }

            self._resource_history[node_id].append(record)

            # 히스토리 크기 제한
            if len(self._resource_history[node_id]) > self._max_history_size:
                self._resource_history[node_id].pop(0)

    def get_resource_history(
        self,
        node_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """리소스 사용 이력 조회."""
        with self._lock:
            history = self._resource_history.get(node_id, [])
            return history[-limit:]

    def get_peak_usage(self, node_id: str) -> dict | None:
        """최고 사용량."""
        with self._lock:
            history = self._resource_history.get(node_id, [])

            if not history:
                return None

            peak_cpu = max(h["cpu_percent"] for h in history)
            peak_memory = max(h["memory_percent"] for h in history)

            return {
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory,
            }

    def get_average_usage(self, node_id: str) -> dict | None:
        """평균 사용량."""
        with self._lock:
            history = self._resource_history.get(node_id, [])

            if not history:
                return None

            avg_cpu = sum(h["cpu_percent"] for h in history) / len(history)
            avg_memory = sum(h["memory_percent"] for h in history) / len(history)

            return {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
            }
