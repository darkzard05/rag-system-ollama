"""
Rollback System for RAG System Version Rollback and Recovery
"""

import json
import time
import logging
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
import hashlib


class RollbackStatus(Enum):
    """Rollback Status Types"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class CheckpointType(Enum):
    """Types of Checkpoints"""

    AUTO = "auto"
    MANUAL = "manual"
    PRE_DEPLOYMENT = "pre_deployment"
    POST_DEPLOYMENT = "post_deployment"


@dataclass
class Checkpoint:
    """System Checkpoint for Recovery"""

    checkpoint_id: str
    name: str
    checkpoint_type: CheckpointType
    version: str
    timestamp: float
    size_bytes: int = 0
    location: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["checkpoint_type"] = self.checkpoint_type.value
        return data


@dataclass
class RollbackRecord:
    """Record of a Rollback Operation"""

    rollback_id: str
    from_version: str
    to_version: str
    status: str
    timestamp: float
    duration: float = 0.0
    checkpoints_used: int = 0
    data_loss_estimate: int = 0  # milliseconds of data loss
    error_message: Optional[str] = None
    initiated_by: str = "system"
    reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RecoveryPlan:
    """Plan for Recovery Operation"""

    plan_id: str
    target_version: str
    target_checkpoint: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: float = 0.0
    estimated_data_loss: int = 0
    risk_level: str = "low"  # low, medium, high
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class RollbackSystem:
    """Manages rollback and recovery operations"""

    def __init__(
        self, checkpoint_root: str = "./checkpoints", retention_days: int = 30
    ):
        """
        Initialize Rollback System

        Args:
            checkpoint_root: Root directory for checkpoints
            retention_days: Days to retain old checkpoints
        """
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        self.checkpoints: Dict[str, Checkpoint] = {}
        self.rollback_history: List[RollbackRecord] = []
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.current_version: str = "1.0.0"
        self.retention_days = retention_days

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

        # Load checkpoints and history
        self._load_checkpoints()
        self._load_rollback_history()

    def _generate_checkpoint_id(self, name: str) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"chk-{timestamp}-{name_hash}"

    def _generate_rollback_id(self) -> str:
        """Generate unique rollback ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        return f"rbk-{timestamp}-{random_suffix}"

    def create_checkpoint(
        self,
        name: str,
        version: str,
        checkpoint_type: CheckpointType,
        data_path: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
    ) -> str:
        """
        Create a system checkpoint

        Args:
            name: Checkpoint name
            version: Version at checkpoint
            checkpoint_type: Type of checkpoint
            data_path: Path to backup data (optional)
            description: Checkpoint description
            created_by: Creator identifier

        Returns:
            Checkpoint ID
        """
        with self._lock:
            checkpoint_id = self._generate_checkpoint_id(name)

            # Create checkpoint directory
            checkpoint_dir = self.checkpoint_root / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            size_bytes = 0

            # Copy data if provided
            if data_path:
                source = Path(data_path)
                if source.exists():
                    if source.is_file():
                        dest = checkpoint_dir / source.name
                        shutil.copy2(source, dest)
                        size_bytes = dest.stat().st_size
                    else:
                        # Copy directory
                        dest = checkpoint_dir / source.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(source, dest)
                        for item in dest.rglob("*"):
                            if item.is_file():
                                size_bytes += item.stat().st_size

            # Create checkpoint metadata
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                name=name,
                checkpoint_type=checkpoint_type,
                version=version,
                timestamp=time.time(),
                size_bytes=size_bytes,
                location=str(checkpoint_dir),
                description=description,
                created_by=created_by,
            )

            self.checkpoints[checkpoint_id] = checkpoint

            # Save checkpoint metadata
            self._save_checkpoint_metadata(checkpoint)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            self.logger.info(
                f"Created checkpoint {checkpoint_id}: {name} for version {version}"
            )

            return checkpoint_id

    def list_checkpoints(
        self,
        version: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available checkpoints

        Args:
            version: Filter by version (optional)
            checkpoint_type: Filter by type (optional)

        Returns:
            List of checkpoints
        """
        with self._lock:
            checkpoints = list(self.checkpoints.values())

            if version:
                checkpoints = [c for c in checkpoints if c.version == version]

            if checkpoint_type:
                checkpoints = [
                    c for c in checkpoints if c.checkpoint_type == checkpoint_type
                ]

            # Sort by timestamp descending
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)

            return [c.to_dict() for c in checkpoints]

    def create_recovery_plan(
        self, target_version: str, target_checkpoint_id: str
    ) -> str:
        """
        Create a recovery plan for rollback

        Args:
            target_version: Target version to recover
            target_checkpoint_id: Checkpoint to use for recovery

        Returns:
            Recovery plan ID
        """
        with self._lock:
            if target_checkpoint_id not in self.checkpoints:
                raise ValueError(f"Checkpoint {target_checkpoint_id} not found")

            checkpoint = self.checkpoints[target_checkpoint_id]

            # Create recovery steps
            steps = [
                {
                    "step": 1,
                    "action": "validate_checkpoint",
                    "checkpoint_id": target_checkpoint_id,
                },
                {
                    "step": 2,
                    "action": "backup_current_state",
                    "version": self.current_version,
                },
                {"step": 3, "action": "stop_services"},
                {
                    "step": 4,
                    "action": "restore_data",
                    "checkpoint_id": target_checkpoint_id,
                },
                {"step": 5, "action": "verify_integrity"},
                {"step": 6, "action": "start_services", "version": target_version},
                {"step": 7, "action": "health_check"},
            ]

            # Calculate estimated data loss
            time_diff = int((time.time() - checkpoint.timestamp) * 1000)

            # Assess risk level
            risk_level = "low"
            if time_diff > 3600000:  # > 1 hour
                risk_level = "medium"
            if time_diff > 86400000:  # > 1 day
                risk_level = "high"

            plan = RecoveryPlan(
                plan_id=f"plan-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                target_version=target_version,
                target_checkpoint=target_checkpoint_id,
                steps=steps,
                estimated_duration=300.0,  # 5 minutes
                estimated_data_loss=time_diff,
                risk_level=risk_level,
            )

            self.recovery_plans[plan.plan_id] = plan

            self.logger.info(
                f"Created recovery plan {plan.plan_id} to version {target_version}"
            )

            return plan.plan_id

    def validate_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Validate a recovery plan before execution

        Args:
            plan_id: Recovery plan ID

        Returns:
            Validation result
        """
        with self._lock:
            if plan_id not in self.recovery_plans:
                raise ValueError(f"Recovery plan {plan_id} not found")

            plan = self.recovery_plans[plan_id]

            validation = {
                "plan_id": plan_id,
                "valid": True,
                "issues": [],
                "warnings": [],
            }

            # Check checkpoint exists
            if plan.target_checkpoint not in self.checkpoints:
                validation["valid"] = False
                validation["issues"].append(
                    f"Target checkpoint {plan.target_checkpoint} not found"
                )

            # Check checkpoint data
            checkpoint = self.checkpoints.get(plan.target_checkpoint)
            if checkpoint and checkpoint.size_bytes == 0:
                validation["warnings"].append("Checkpoint has no data")

            # Check data loss estimate
            if plan.estimated_data_loss > 3600000:  # > 1 hour
                validation["warnings"].append(
                    f"Potential data loss: {plan.estimated_data_loss}ms"
                )

            return validation

    def execute_recovery_plan(
        self, plan_id: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute recovery plan

        Args:
            plan_id: Recovery plan ID
            dry_run: If True, validate but don't execute

        Returns:
            Execution result
        """
        with self._lock:
            if plan_id not in self.recovery_plans:
                raise ValueError(f"Recovery plan {plan_id} not found")

            # Validate first
            validation = self.validate_recovery_plan(plan_id)
            if not validation["valid"]:
                raise RuntimeError(
                    f"Recovery plan validation failed: {validation['issues']}"
                )

            plan = self.recovery_plans[plan_id]

            if dry_run:
                return {
                    "plan_id": plan_id,
                    "status": "validated",
                    "dry_run": True,
                    "message": "Recovery plan passed validation (dry run)",
                }

            rollback_id = self._generate_rollback_id()
            start_time = time.time()

            try:
                # Execute recovery steps (simulated)
                for step in plan.steps:
                    step_action = step["action"]

                    if step_action == "validate_checkpoint":
                        # Validate checkpoint integrity
                        pass
                    elif step_action == "backup_current_state":
                        # Backup current state before recovery
                        pass
                    elif step_action == "stop_services":
                        # Stop running services
                        pass
                    elif step_action == "restore_data":
                        # Restore data from checkpoint
                        pass
                    elif step_action == "verify_integrity":
                        # Verify restored data integrity
                        pass
                    elif step_action == "start_services":
                        # Start services with new version
                        pass
                    elif step_action == "health_check":
                        # Perform health checks
                        pass

                # Create rollback record
                record = RollbackRecord(
                    rollback_id=rollback_id,
                    from_version=self.current_version,
                    to_version=plan.target_version,
                    status=RollbackStatus.COMPLETED.value,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    checkpoints_used=1,
                    data_loss_estimate=plan.estimated_data_loss,
                    initiated_by="system",
                    reason="recovery",
                )

                self.rollback_history.append(record)

                # Update current version
                self.current_version = plan.target_version

                # Save history
                self._save_rollback_history()

                self.logger.info(
                    f"Executed recovery plan {plan_id} in {record.duration:.2f}s"
                )

                return {
                    "rollback_id": rollback_id,
                    "status": "completed",
                    "duration": record.duration,
                    "data_loss_estimate_ms": plan.estimated_data_loss,
                }

            except Exception as e:
                # Record failure
                record = RollbackRecord(
                    rollback_id=rollback_id,
                    from_version=self.current_version,
                    to_version=plan.target_version,
                    status=RollbackStatus.FAILED.value,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    error_message=str(e),
                    reason="recovery",
                )

                self.rollback_history.append(record)
                self._save_rollback_history()

                self.logger.error(
                    f"Failed to execute recovery plan {plan_id}: {str(e)}"
                )

                raise

    def manual_rollback(
        self, target_version: str, checkpoint_id: Optional[str] = None, reason: str = ""
    ) -> Dict[str, Any]:
        """
        Perform manual rollback

        Args:
            target_version: Target version to rollback to
            checkpoint_id: Specific checkpoint to use (optional)
            reason: Reason for rollback

        Returns:
            Rollback result
        """
        with self._lock:
            # Find latest checkpoint for target version
            if checkpoint_id is None:
                target_checkpoints = [
                    c for c in self.checkpoints.values() if c.version == target_version
                ]

                if not target_checkpoints:
                    raise ValueError(
                        f"No checkpoints found for version {target_version}"
                    )

                # Use most recent checkpoint
                checkpoint_id = sorted(
                    target_checkpoints, key=lambda x: x.timestamp, reverse=True
                )[0].checkpoint_id

            # Create and execute recovery plan
            plan_id = self.create_recovery_plan(target_version, checkpoint_id)

            return self.execute_recovery_plan(plan_id)

    def get_rollback_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get rollback history

        Args:
            limit: Number of records to return

        Returns:
            List of rollback records
        """
        with self._lock:
            return [r.to_dict() for r in self.rollback_history[-limit:]]

    def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get detailed checkpoint information
        """
        with self._lock:
            if checkpoint_id not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            checkpoint = self.checkpoints[checkpoint_id]
            checkpoint_dir = Path(checkpoint.location)

            # Calculate data details
            file_count = 0
            exists = checkpoint_dir.exists()
            if exists:
                file_count = sum(1 for _ in checkpoint_dir.rglob("*") if _.is_file())

            return {
                **checkpoint.to_dict(),
                "file_count": file_count,
                "exists": exists,
                "age_hours": (time.time() - checkpoint.timestamp) / 3600,
            }

    def sync_checkpoints(self) -> Dict[str, int]:
        """
        Synchronizes the internal registry with the physical file system.
        Removes records for non-existent directories and cleans up un-indexed folders.
        """
        with self._lock:
            removed_records = 0
            # 1. Remove records for missing directories
            ids_to_remove = []
            for cid, checkpoint in self.checkpoints.items():
                if not Path(checkpoint.location).exists():
                    ids_to_remove.append(cid)

            for cid in ids_to_remove:
                del self.checkpoints[cid]
                removed_records += 1

            # 2. Find orphaned directories (chk- folders with no metadata)
            cleaned_dirs = 0
            if self.checkpoint_root.exists():
                for folder in self.checkpoint_root.iterdir():
                    if folder.is_dir() and folder.name.startswith("chk-"):
                        if folder.name not in self.checkpoints:
                            try:
                                shutil.rmtree(folder)
                                cleaned_dirs += 1
                            except:
                                pass

            return {
                "removed_records": removed_records,
                "cleaned_directories": cleaned_dirs,
            }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding retention period"""
        try:
            cutoff_time = time.time() - (self.retention_days * 86400)

            checkpoints_to_remove = [
                checkpoint_id
                for checkpoint_id, checkpoint in self.checkpoints.items()
                if checkpoint.timestamp < cutoff_time
            ]

            for checkpoint_id in checkpoints_to_remove:
                checkpoint = self.checkpoints[checkpoint_id]

                # Remove checkpoint directory
                checkpoint_dir = Path(checkpoint.location)
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)

                # Remove from registry
                del self.checkpoints[checkpoint_id]

                self.logger.info(f"Cleaned up old checkpoint {checkpoint_id}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {str(e)}")

    def _save_checkpoint_metadata(self, checkpoint: Checkpoint):
        """Save checkpoint metadata to file"""
        try:
            metadata_file = Path(checkpoint.location) / "checkpoint.json"

            with open(metadata_file, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint metadata: {str(e)}")

    def _save_rollback_history(self):
        """Save rollback history to file"""
        try:
            history_file = self.checkpoint_root / "rollback_history.json"
            history_data = [r.to_dict() for r in self.rollback_history]

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save rollback history: {str(e)}")

    def _load_checkpoints(self):
        """Load checkpoints from files"""
        try:
            for checkpoint_dir in self.checkpoint_root.iterdir():
                if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("chk-"):
                    metadata_file = checkpoint_dir / "checkpoint.json"

                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            checkpoint_data = json.load(f)
                        # Convert checkpoint_type string back to enum
                        checkpoint_data["checkpoint_type"] = CheckpointType(
                            checkpoint_data["checkpoint_type"]
                        )
                        checkpoint = Checkpoint(**checkpoint_data)
                        self.checkpoints[checkpoint.checkpoint_id] = checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoints: {str(e)}")

    def _load_rollback_history(self):
        """Load rollback history from file"""
        try:
            history_file = self.checkpoint_root / "rollback_history.json"

            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)

                for record_data in history_data:
                    record = RollbackRecord(**record_data)
                    self.rollback_history.append(record)

        except Exception as e:
            self.logger.error(f"Failed to load rollback history: {str(e)}")

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """
        Get rollback statistics

        Returns:
            Statistics about rollbacks
        """
        with self._lock:
            total_checkpoints = len(self.checkpoints)
            total_size_bytes = sum(c.size_bytes for c in self.checkpoints.values())

            rollbacks_completed = sum(
                1
                for r in self.rollback_history
                if r.status == RollbackStatus.COMPLETED.value
            )
            rollbacks_failed = sum(
                1
                for r in self.rollback_history
                if r.status == RollbackStatus.FAILED.value
            )

            # Count by checkpoint type
            by_type = {}
            for checkpoint in self.checkpoints.values():
                checkpoint_type = checkpoint.checkpoint_type.value
                if checkpoint_type not in by_type:
                    by_type[checkpoint_type] = 0
                by_type[checkpoint_type] += 1

            # Average data loss for completed rollbacks
            avg_data_loss = 0
            if rollbacks_completed > 0:
                total_loss = sum(
                    r.data_loss_estimate
                    for r in self.rollback_history
                    if r.status == RollbackStatus.COMPLETED.value
                )
                avg_data_loss = total_loss / rollbacks_completed

            return {
                "total_checkpoints": total_checkpoints,
                "total_checkpoint_size_bytes": total_size_bytes,
                "total_rollbacks": len(self.rollback_history),
                "successful_rollbacks": rollbacks_completed,
                "failed_rollbacks": rollbacks_failed,
                "success_rate": rollbacks_completed / len(self.rollback_history)
                if self.rollback_history
                else 0,
                "average_data_loss_ms": avg_data_loss,
                "checkpoints_by_type": by_type,
                "current_version": self.current_version,
            }
