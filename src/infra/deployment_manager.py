"""
Deployment Manager for RAG System Deployment and Versioning
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class DeploymentStatus(Enum):
    """Deployment Status Types"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentEnvironment(Enum):
    """Deployment Target Environments"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Configuration for Deployment"""

    service_name: str
    version: str
    environment: DeploymentEnvironment
    deployment_path: str
    backup_enabled: bool = True
    health_check_enabled: bool = True
    rollback_on_failure: bool = True
    max_concurrent_deployments: int = 3
    deployment_timeout: int = 300  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["environment"] = self.environment.value
        return data


@dataclass
class DeploymentRecord:
    """Record of a Deployment"""

    deployment_id: str
    service_name: str
    version: str
    environment: str
    status: str
    timestamp: float
    duration: float = 0.0
    checksum: str = ""
    previous_version: str | None = None
    error_message: str | None = None
    deployed_by: str = "system"
    artifacts_hash: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HealthCheckResult:
    """Health Check Result for Deployment"""

    service_name: str
    status: bool
    timestamp: float
    checks: dict[str, bool] = field(default_factory=dict)
    error_details: str | None = None
    response_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class DeploymentManager:
    """Manages deployment of RAG System versions"""

    def __init__(self, deployment_root: str = "./deployments"):
        """
        Initialize Deployment Manager

        Args:
            deployment_root: Root directory for deployments
        """
        self.deployment_root = Path(deployment_root)
        self.deployment_root.mkdir(parents=True, exist_ok=True)

        self.deployments: dict[str, DeploymentRecord] = {}
        self.deployment_history: list[DeploymentRecord] = []
        self.active_deployments: dict[str, DeploymentConfig] = {}
        self.health_checks: dict[str, list[HealthCheckResult]] = {}

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

        # Load deployment history from file
        self._load_deployment_history()

    def _generate_deployment_id(self, service_name: str, version: str) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().isoformat()
        data = f"{service_name}:{version}:{timestamp}".encode()
        hash_value = hashlib.sha256(data).hexdigest()[:12]
        return f"dep-{hash_value}"

    def _calculate_checksum(self, deployment_path: str) -> str:
        """Calculate checksum for deployment artifacts"""
        sha256_hash = hashlib.sha256()

        path = Path(deployment_path)
        if not path.exists():
            return ""

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        else:
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    with open(file, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def start_deployment(self, config: DeploymentConfig) -> str:
        """
        Start deployment process

        Args:
            config: Deployment configuration

        Returns:
            Deployment ID
        """
        with self._lock:
            # Check concurrent deployment limit
            if len(self.active_deployments) >= config.max_concurrent_deployments:
                raise RuntimeError(
                    f"Max concurrent deployments ({config.max_concurrent_deployments}) reached"
                )

            deployment_id = self._generate_deployment_id(
                config.service_name, config.version
            )

            # Check if version already deployed
            for record in self.deployment_history:
                if (
                    record.service_name == config.service_name
                    and record.version == config.version
                    and record.status == DeploymentStatus.COMPLETED.value
                ):
                    raise ValueError(
                        f"Version {config.version} already deployed for {config.service_name}"
                    )

            # Get previous version
            previous_version = None
            for record in reversed(self.deployment_history):
                if (
                    record.service_name == config.service_name
                    and record.status == DeploymentStatus.COMPLETED.value
                ):
                    previous_version = record.version
                    break

            # Create deployment directory
            deployment_dir = self.deployment_root / deployment_id
            deployment_dir.mkdir(parents=True, exist_ok=True)

            # Create deployment record
            record = DeploymentRecord(
                deployment_id=deployment_id,
                service_name=config.service_name,
                version=config.version,
                environment=config.environment.value,
                status=DeploymentStatus.IN_PROGRESS.value,
                timestamp=time.time(),
                previous_version=previous_version,
                deployed_by=config.metadata.get("deployed_by", "system"),
            )

            self.deployments[deployment_id] = record
            self.active_deployments[deployment_id] = config

            self.logger.info(
                f"Started deployment: {deployment_id} for {config.service_name} v{config.version}"
            )

            return deployment_id

    def deploy_artifacts(self, deployment_id: str, source_path: str) -> bool:
        """
        Deploy artifacts to deployment directory with automatic cleanup on failure.
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")

            record = self.deployments[deployment_id]
            if record.status != DeploymentStatus.IN_PROGRESS.value:
                raise RuntimeError(f"Deployment {deployment_id} is not in progress")

            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source path not found: {source_path}")

            # Copy artifacts to deployment directory
            deployment_dir = self.deployment_root / deployment_id / "artifacts"
            deployment_dir.mkdir(parents=True, exist_ok=True)

            try:
                if source.is_file():
                    shutil.copy2(source, deployment_dir)
                else:
                    # Copy directory contents
                    for item in source.iterdir():
                        if item.is_file():
                            shutil.copy2(item, deployment_dir)
                        elif item.is_dir():
                            dest = deployment_dir / item.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)

                # Calculate checksum
                record.checksum = self._calculate_checksum(str(deployment_dir))

                self.logger.info(
                    f"Deployed artifacts to {deployment_id}, checksum: {record.checksum}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Failed to deploy artifacts: {str(e)}")
                # [Resource Leak Prevention] Clean up the partially copied artifacts
                if deployment_dir.exists():
                    shutil.rmtree(deployment_dir)
                    self.logger.info(
                        f"Cleaned up partial artifacts for failed deployment {deployment_id}"
                    )
                raise

    def purge_failed_deployments(self) -> int:
        """
        Removes physical directories for all deployments marked as FAILED.
        Returns the number of directories removed.
        """
        with self._lock:
            count = 0
            for deployment_id, record in self.deployments.items():
                if record.status == DeploymentStatus.FAILED.value:
                    dep_dir = self.deployment_root / deployment_id
                    if dep_dir.exists():
                        try:
                            shutil.rmtree(dep_dir)
                            count += 1
                            self.logger.info(
                                f"Purged failed deployment directory: {deployment_id}"
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to purge {deployment_id}: {e}")
            return count

    def health_check(
        self, deployment_id: str, checks: dict[str, bool] | None = None
    ) -> HealthCheckResult:
        """
        Perform health check on deployment

        Args:
            deployment_id: Deployment ID
            checks: Dictionary of health checks

        Returns:
            Health check result
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")

            record = self.deployments[deployment_id]

            # Default health checks
            if checks is None:
                checks = {
                    "artifacts_exist": (
                        self.deployment_root / deployment_id / "artifacts"
                    ).exists(),
                    "checksum_valid": bool(record.checksum),
                    "configuration_valid": deployment_id in self.active_deployments,
                }

            # Perform checks
            all_passed = True
            check_results = {}
            for check_name, result in checks.items():
                check_results[check_name] = result
                if not result:
                    all_passed = False

            health_result = HealthCheckResult(
                service_name=record.service_name,
                status=all_passed,
                timestamp=time.time(),
                checks=check_results,
                response_time_ms=1.0,  # Simulated response time
            )

            # Store health check result
            if deployment_id not in self.health_checks:
                self.health_checks[deployment_id] = []
            self.health_checks[deployment_id].append(health_result)

            self.logger.info(f"Health check for {deployment_id}: {all_passed}")

            return health_result

    def complete_deployment(self, deployment_id: str) -> bool:
        """
        Complete deployment process

        Args:
            deployment_id: Deployment ID

        Returns:
            Success status
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")

            record = self.deployments[deployment_id]
            if record.status != DeploymentStatus.IN_PROGRESS.value:
                raise RuntimeError(f"Deployment {deployment_id} is not in progress")

            # Update record
            record.status = DeploymentStatus.COMPLETED.value
            record.duration = time.time() - record.timestamp

            # Add to history
            self.deployment_history.append(record)

            # Remove from active deployments
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

            # Save deployment history
            self._save_deployment_history()

            self.logger.info(
                f"Completed deployment {deployment_id} in {record.duration:.2f}s"
            )

            return True

    def fail_deployment(self, deployment_id: str, error_message: str) -> bool:
        """
        Mark deployment as failed

        Args:
            deployment_id: Deployment ID
            error_message: Error message

        Returns:
            Success status
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")

            record = self.deployments[deployment_id]
            record.status = DeploymentStatus.FAILED.value
            record.error_message = error_message
            record.duration = time.time() - record.timestamp

            # Add to history
            self.deployment_history.append(record)

            # Remove from active deployments
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

            # Save deployment history
            self._save_deployment_history()

            self.logger.error(f"Failed deployment {deployment_id}: {error_message}")

            return True

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """
        Get deployment status

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status information
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")

            record = self.deployments[deployment_id]
            health_check_list = self.health_checks.get(deployment_id, [])

            return {
                "deployment_id": deployment_id,
                "status": record.status,
                "service_name": record.service_name,
                "version": record.version,
                "environment": record.environment,
                "timestamp": record.timestamp,
                "duration": record.duration,
                "checksum": record.checksum,
                "previous_version": record.previous_version,
                "health_checks": [
                    hc.to_dict() for hc in health_check_list[-5:]
                ],  # Last 5
                "error_message": record.error_message,
            }

    def get_deployment_history(
        self, service_name: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get deployment history

        Args:
            service_name: Filter by service name (optional)
            limit: Number of records to return

        Returns:
            List of deployment records
        """
        with self._lock:
            history = self.deployment_history

            if service_name:
                history = [r for r in history if r.service_name == service_name]

            # Return most recent
            return [r.to_dict() for r in history[-limit:]]

    def get_active_deployments(self) -> list[dict[str, Any]]:
        """
        Get active deployments

        Returns:
            List of active deployment configurations
        """
        with self._lock:
            return [
                {
                    "deployment_id": dep_id,
                    "service_name": config.service_name,
                    "version": config.version,
                    "environment": config.environment.value,
                    "started": self.deployments[dep_id].timestamp,
                }
                for dep_id, config in self.active_deployments.items()
            ]

    def _save_deployment_history(self):
        """Save deployment history to file"""
        try:
            history_file = self.deployment_root / "deployment_history.json"
            history_data = [r.to_dict() for r in self.deployment_history]

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save deployment history: {str(e)}")

    def _load_deployment_history(self):
        """Load deployment history from file"""
        try:
            history_file = self.deployment_root / "deployment_history.json"

            if history_file.exists():
                with open(history_file) as f:
                    history_data = json.load(f)

                for record_data in history_data:
                    record = DeploymentRecord(**record_data)
                    self.deployment_history.append(record)

        except Exception as e:
            self.logger.error(f"Failed to load deployment history: {str(e)}")

    def get_deployment_statistics(self) -> dict[str, Any]:
        """
        Get deployment statistics

        Returns:
            Statistics about deployments
        """
        with self._lock:
            total_deployments = len(self.deployment_history)
            successful = sum(
                1
                for r in self.deployment_history
                if r.status == DeploymentStatus.COMPLETED.value
            )
            failed = sum(
                1
                for r in self.deployment_history
                if r.status == DeploymentStatus.FAILED.value
            )

            # Calculate average duration for successful deployments
            successful_deployments = [
                r
                for r in self.deployment_history
                if r.status == DeploymentStatus.COMPLETED.value
            ]
            avg_duration = (
                sum(r.duration for r in successful_deployments)
                / len(successful_deployments)
                if successful_deployments
                else 0
            )

            # Service breakdown
            services = {}
            for record in self.deployment_history:
                if record.service_name not in services:
                    services[record.service_name] = {"total": 0, "successful": 0}
                services[record.service_name]["total"] += 1
                if record.status == DeploymentStatus.COMPLETED.value:
                    services[record.service_name]["successful"] += 1

            return {
                "total_deployments": total_deployments,
                "successful_deployments": successful,
                "failed_deployments": failed,
                "success_rate": successful / total_deployments
                if total_deployments > 0
                else 0,
                "average_duration_seconds": avg_duration,
                "active_deployments": len(self.active_deployments),
                "services": services,
            }

    def cleanup_orphaned_artifacts(
        self, max_age_hours: int = 24, target_dir: Path | None = None
    ) -> int:
        """
        [리소스 정리] 생성된 지 오래된 고스트 디렉토리나 임시 파일을 정리합니다.

        Args:
            max_age_hours: 삭제 기준 시간 (시간 단위)
            target_dir: 정리할 대상 디렉토리 (None이면 deployment_root 사용)

        Returns:
            삭제된 항목 수
        """
        with self._lock:
            count = 0
            now = time.time()
            max_age_seconds = max_age_hours * 3600

            root = target_dir if target_dir is not None else self.deployment_root

            if not root.exists():
                return 0

            for item in root.iterdir():
                # 활성 배포 중인 폴더는 제외 (target_dir이 deployment_root일 때만 유효)
                if target_dir is None and item.name in self.active_deployments:
                    continue

                # 폴더 또는 파일의 수정 시간 확인
                mtime = item.stat().st_mtime
                if (now - mtime) > max_age_seconds:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        count += 1
                        self.logger.info(f"Cleaned up orphaned artifact: {item.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup {item.name}: {e}")

            return count


# [추가] 전역 싱글톤 인스턴스 관리
_deployment_manager_instance = None
_instance_lock = RLock()


def get_deployment_manager() -> "DeploymentManager":
    """DeploymentManager의 전역 싱글톤 인스턴스를 반환합니다."""
    global _deployment_manager_instance
    if _deployment_manager_instance is None:
        with _instance_lock:
            if _deployment_manager_instance is None:
                _deployment_manager_instance = DeploymentManager()
    return _deployment_manager_instance
