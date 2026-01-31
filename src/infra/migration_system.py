"""
Migration System for RAG System Database and Schema Migrations
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class MigrationStatus(Enum):
    """Migration Status Types"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationType(Enum):
    """Types of Migrations"""

    SCHEMA = "schema"
    DATA = "data"
    CONFIG = "config"
    INDEX = "index"


@dataclass
class MigrationScript:
    """Database Migration Script"""

    migration_id: str
    name: str
    migration_type: MigrationType
    version: str
    script: str  # SQL or code
    checksum: str = ""
    dependencies: list[str] = field(default_factory=list)
    rollback_script: str | None = None
    description: str = ""
    author: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["migration_type"] = self.migration_type.value
        return data


@dataclass
class MigrationRecord:
    """Record of a Completed Migration"""

    migration_id: str
    name: str
    version: str
    status: str
    timestamp: float
    duration: float = 0.0
    affected_records: int = 0
    error_message: str | None = None
    executed_by: str = "system"
    rows_changed: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SchemaVersion:
    """Schema Version Information"""

    version: str
    timestamp: float
    description: str
    tables: dict[str, list[str]] = field(default_factory=dict)  # table -> columns
    indexes: dict[str, list[str]] = field(default_factory=dict)  # table -> indexes

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class MigrationSystem:
    """Manages database and schema migrations"""

    def __init__(self, migration_root: str = "./migrations", db_connection=None):
        """
        Initialize Migration System

        Args:
            migration_root: Root directory for migrations
            db_connection: Database connection (mock for testing)
        """
        self.migration_root = Path(migration_root)
        self.migration_root.mkdir(parents=True, exist_ok=True)

        self.migrations: dict[str, MigrationScript] = {}
        self.migration_history: list[MigrationRecord] = []
        self.current_schema_version: SchemaVersion | None = None
        self.db_connection = db_connection

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

        # Load migrations and history
        self._load_migrations()
        self._load_migration_history()

    def _generate_migration_id(self, name: str) -> str:
        """Generate unique migration ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{name}"

    def _calculate_checksum(self, script: str) -> str:
        """Calculate checksum for migration script"""
        return hashlib.sha256(script.encode()).hexdigest()

    def create_migration(
        self,
        name: str,
        script: str,
        migration_type: MigrationType,
        rollback_script: str | None = None,
        dependencies: list[str] | None = None,
        description: str = "",
        author: str = "system",
    ) -> str:
        """
        Create a new migration

        Args:
            name: Migration name
            script: Migration script (SQL or code)
            migration_type: Type of migration
            rollback_script: Rollback script (optional)
            dependencies: List of migration IDs this depends on
            description: Migration description
            author: Author of migration

        Returns:
            Migration ID
        """
        with self._lock:
            version = datetime.now().strftime("%Y.%m.%d.%H%M")
            migration_id = self._generate_migration_id(name)

            migration = MigrationScript(
                migration_id=migration_id,
                name=name,
                migration_type=migration_type,
                version=version,
                script=script,
                checksum=self._calculate_checksum(script),
                rollback_script=rollback_script,
                dependencies=dependencies or [],
                description=description,
                author=author,
            )

            self.migrations[migration_id] = migration

            # Save migration to file
            self._save_migration(migration)

            self.logger.info(f"Created migration {migration_id}: {name}")

            return migration_id

    def validate_migration(self, migration_id: str) -> dict[str, Any]:
        """
        Validate migration before execution

        Args:
            migration_id: Migration ID

        Returns:
            Validation result
        """
        with self._lock:
            if migration_id not in self.migrations:
                raise ValueError(f"Migration {migration_id} not found")

            migration = self.migrations[migration_id]
            validation_result: dict[str, Any] = {
                "migration_id": migration_id,
                "valid": True,
                "issues": [],
            }

            # Check if already executed
            for record in self.migration_history:
                if (
                    record.migration_id == migration_id
                    and record.status == MigrationStatus.COMPLETED.value
                ):
                    validation_result["valid"] = False
                    validation_result["issues"].append("Already executed")
                    break

            # Check dependencies
            for dep_id in migration.dependencies:
                if dep_id not in self.migrations:
                    validation_result["valid"] = False
                    validation_result["issues"].append(f"Dependency {dep_id} not found")
                else:
                    # Check if dependency is executed
                    dep_executed = any(
                        r.migration_id == dep_id
                        and r.status == MigrationStatus.COMPLETED.value
                        for r in self.migration_history
                    )
                    if not dep_executed:
                        validation_result["valid"] = False
                        validation_result["issues"].append(
                            f"Dependency {dep_id} not executed"
                        )

            # Check script validity
            if not migration.script or len(migration.script.strip()) == 0:
                validation_result["valid"] = False
                validation_result["issues"].append("Migration script is empty")

            return validation_result

    def execute_migration(
        self, migration_id: str, dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Execute migration

        Args:
            migration_id: Migration ID
            dry_run: If True, validate but don't execute

        Returns:
            Execution result
        """
        with self._lock:
            if migration_id not in self.migrations:
                raise ValueError(f"Migration {migration_id} not found")

            # Validate first
            validation = self.validate_migration(migration_id)
            if not validation["valid"]:
                raise RuntimeError(
                    f"Migration validation failed: {validation['issues']}"
                )

            migration = self.migrations[migration_id]

            if dry_run:
                result: dict[str, Any] = {
                    "migration_id": migration_id,
                    "status": "validated",
                    "dry_run": True,
                    "message": "Migration passed validation (dry run)",
                }
                return result

            start_time = time.time()

            try:
                # Execute migration script (simulated)
                affected_records = 0
                rows_changed = 0

                if migration.migration_type == MigrationType.SCHEMA:
                    # Simulate schema change
                    affected_records = 1
                elif migration.migration_type == MigrationType.DATA:
                    # Simulate data migration
                    affected_records = 100
                    rows_changed = 100
                elif migration.migration_type == MigrationType.INDEX:
                    # Simulate index creation
                    affected_records = 1
                elif migration.migration_type == MigrationType.CONFIG:
                    # Simulate config migration
                    affected_records = 1

                # Create migration record
                record = MigrationRecord(
                    migration_id=migration_id,
                    name=migration.name,
                    version=migration.version,
                    status=MigrationStatus.COMPLETED.value,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    affected_records=affected_records,
                    rows_changed=rows_changed,
                )

                self.migration_history.append(record)

                # Update current schema version
                if migration.migration_type == MigrationType.SCHEMA:
                    self.current_schema_version = SchemaVersion(
                        version=migration.version,
                        timestamp=time.time(),
                        description=migration.description,
                    )

                # Save history
                self._save_migration_history()

                self.logger.info(
                    f"Executed migration {migration_id} in {record.duration:.2f}s"
                )

                return {
                    "migration_id": migration_id,
                    "status": "completed",
                    "duration": record.duration,
                    "affected_records": affected_records,
                    "rows_changed": rows_changed,
                }

            except Exception as e:
                # Record failure
                record = MigrationRecord(
                    migration_id=migration_id,
                    name=migration.name,
                    version=migration.version,
                    status=MigrationStatus.FAILED.value,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    error_message=str(e),
                )

                self.migration_history.append(record)
                self._save_migration_history()

                self.logger.error(
                    f"Failed to execute migration {migration_id}: {str(e)}"
                )

                raise

    def rollback_migration(self, migration_id: str) -> dict[str, Any]:
        """
        Rollback a migration

        Args:
            migration_id: Migration ID

        Returns:
            Rollback result
        """
        with self._lock:
            if migration_id not in self.migrations:
                raise ValueError(f"Migration {migration_id} not found")

            migration = self.migrations[migration_id]

            # Check if migration was executed
            executed_record = None
            for record in self.migration_history:
                if (
                    record.migration_id == migration_id
                    and record.status == MigrationStatus.COMPLETED.value
                ):
                    executed_record = record
                    break

            if not executed_record:
                raise RuntimeError(f"Migration {migration_id} was not executed")

            if not migration.rollback_script:
                raise RuntimeError(f"Migration {migration_id} has no rollback script")

            start_time = time.time()

            try:
                # Execute rollback script (simulated)
                affected_records = executed_record.affected_records
                rows_changed = executed_record.rows_changed

                # Update history record
                executed_record.status = MigrationStatus.ROLLED_BACK.value

                # Create new record for rollback
                rollback_record = MigrationRecord(
                    migration_id=f"{migration_id}_rollback",
                    name=f"{migration.name} (rollback)",
                    version=migration.version,
                    status=MigrationStatus.COMPLETED.value,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    affected_records=affected_records,
                    rows_changed=rows_changed,
                )

                self.migration_history.append(rollback_record)
                self._save_migration_history()

                self.logger.info(
                    f"Rolled back migration {migration_id} in {rollback_record.duration:.2f}s"
                )

                return {
                    "migration_id": migration_id,
                    "status": "rolled_back",
                    "duration": rollback_record.duration,
                    "affected_records": affected_records,
                }

            except Exception as e:
                self.logger.error(
                    f"Failed to rollback migration {migration_id}: {str(e)}"
                )
                raise

    def get_migration_status(self, migration_id: str) -> dict[str, Any]:
        """
        Get migration status

        Args:
            migration_id: Migration ID

        Returns:
            Migration status
        """
        with self._lock:
            if migration_id not in self.migrations:
                raise ValueError(f"Migration {migration_id} not found")

            migration = self.migrations[migration_id]

            # Find latest execution
            latest_execution = None
            for record in reversed(self.migration_history):
                if record.migration_id == migration_id:
                    latest_execution = record
                    break

            return {
                "migration_id": migration_id,
                "name": migration.name,
                "type": migration.migration_type.value,
                "version": migration.version,
                "status": latest_execution.status if latest_execution else "pending",
                "last_executed": latest_execution.timestamp
                if latest_execution
                else None,
                "duration": latest_execution.duration if latest_execution else 0,
                "dependencies": migration.dependencies,
                "description": migration.description,
            }

    def get_migration_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get migration history

        Args:
            limit: Number of records to return

        Returns:
            List of migration records
        """
        with self._lock:
            return [r.to_dict() for r in self.migration_history[-limit:]]

    def get_pending_migrations(self) -> list[dict[str, Any]]:
        """
        Get pending migrations

        Returns:
            List of migrations not yet executed
        """
        with self._lock:
            executed_ids = {
                r.migration_id
                for r in self.migration_history
                if r.status == MigrationStatus.COMPLETED.value
            }

            pending = [
                {
                    "migration_id": m_id,
                    "name": m.name,
                    "type": m.migration_type.value,
                    "version": m.version,
                    "dependencies": m.dependencies,
                }
                for m_id, m in self.migrations.items()
                if m_id not in executed_ids
            ]

            return pending

    def check_schema_compatibility(self, target_version: str) -> dict[str, Any]:
        """
        Check schema compatibility between versions

        Args:
            target_version: Target schema version

        Returns:
            Compatibility information
        """
        with self._lock:
            current_version = (
                self.current_schema_version.version
                if self.current_schema_version
                else "1.0.0"
            )

            # Parse versions
            current_parts = [int(x) for x in current_version.split(".")]
            target_parts = [int(x) for x in target_version.split(".")]

            major_change = current_parts[0] != target_parts[0]

            return {
                "current_version": current_version,
                "target_version": target_version,
                "compatible": bool(not major_change),
                "requires_migration": current_version != target_version,
                "breaking_changes": major_change,
                "migrations_required": [
                    m
                    for m in self.get_pending_migrations()
                    if m["version"] > current_version and m["version"] <= target_version
                ],
            }

    def get_schema_version(self) -> dict[str, Any]:
        """
        Get current schema version

        Returns:
            Schema version information
        """
        with self._lock:
            if self.current_schema_version:
                return self.current_schema_version.to_dict()
            else:
                return {
                    "version": "1.0.0",
                    "timestamp": time.time(),
                    "description": "Initial schema",
                    "tables": {},
                    "indexes": {},
                }

    def _save_migration(self, migration: MigrationScript):
        """Save migration script to file"""
        try:
            migration_dir = self.migration_root / migration.migration_type.value
            migration_dir.mkdir(parents=True, exist_ok=True)

            migration_file = migration_dir / f"{migration.migration_id}.json"

            with open(migration_file, "w") as f:
                json.dump(migration.to_dict(), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save migration: {str(e)}")

    def _save_migration_history(self):
        """Save migration history to file"""
        try:
            history_file = self.migration_root / "migration_history.json"
            history_data = [r.to_dict() for r in self.migration_history]

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save migration history: {str(e)}")

    def _load_migrations(self):
        """Load migrations from files"""
        try:
            for migration_type in MigrationType:
                migration_dir = self.migration_root / migration_type.value
                if migration_dir.exists():
                    for migration_file in migration_dir.glob("*.json"):
                        with open(migration_file) as f:
                            migration_data = json.load(f)
                        # Convert migration_type string back to enum
                        migration_data["migration_type"] = MigrationType(
                            migration_data["migration_type"]
                        )
                        migration = MigrationScript(**migration_data)
                        self.migrations[migration.migration_id] = migration

        except Exception as e:
            self.logger.error(f"Failed to load migrations: {str(e)}")

    def _load_migration_history(self):
        """Load migration history from file"""
        try:
            history_file = self.migration_root / "migration_history.json"

            if history_file.exists():
                with open(history_file) as f:
                    history_data = json.load(f)

                for record_data in history_data:
                    record = MigrationRecord(**record_data)
                    self.migration_history.append(record)

        except Exception as e:
            self.logger.error(f"Failed to load migration history: {str(e)}")

    def get_migration_statistics(self) -> dict[str, Any]:
        """
        Get migration statistics

        Returns:
            Statistics about migrations
        """
        with self._lock:
            total_migrations = len(self.migrations)
            executed = sum(
                1
                for r in self.migration_history
                if r.status == MigrationStatus.COMPLETED.value
            )
            failed = sum(
                1
                for r in self.migration_history
                if r.status == MigrationStatus.FAILED.value
            )
            rolled_back = sum(
                1
                for r in self.migration_history
                if r.status == MigrationStatus.ROLLED_BACK.value
            )

            # Count by type
            by_type = {}
            for migration in self.migrations.values():
                migration_type = migration.migration_type.value
                if migration_type not in by_type:
                    by_type[migration_type] = 0
                by_type[migration_type] += 1

            return {
                "total_migrations": total_migrations,
                "executed_migrations": executed,
                "failed_migrations": failed,
                "rolled_back_migrations": rolled_back,
                "pending_migrations": len(self.get_pending_migrations()),
                "success_rate": executed / total_migrations
                if total_migrations > 0
                else 0,
                "migrations_by_type": by_type,
            }
