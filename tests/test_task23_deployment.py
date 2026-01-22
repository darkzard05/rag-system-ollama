"""
Test Suite for Task 23: Deployment, Migration, and Rollback System
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from src.infra.deployment_manager import (
    DeploymentManager, DeploymentConfig, DeploymentEnvironment,
    DeploymentStatus
)
from src.infra.migration_system import (
    MigrationSystem, MigrationType
)
from src.infra.rollback_system import (
    RollbackSystem, CheckpointType, RollbackStatus
)


class TestDeploymentSystem:
    """Test Deployment Manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.deployment_root = f"{self.temp_dir}/deployments"
        self.manager = DeploymentManager(deployment_root=self.deployment_root)
    
    def teardown_method(self):
        """Cleanup test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_01_create_deployment_config(self):
        """Test creating deployment configuration"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.PRODUCTION,
            deployment_path="/opt/rag",
            backup_enabled=True,
            health_check_enabled=True,
        )
        
        assert config.service_name == "rag-core"
        assert config.version == "2.0.0"
        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.backup_enabled is True
    
    def test_02_start_deployment(self):
        """Test starting deployment"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        
        assert deployment_id is not None
        assert deployment_id.startswith("dep-")
        assert deployment_id in self.manager.deployments
    
    def test_03_deploy_artifacts(self):
        """Test deploying artifacts"""
        # Create test artifact
        artifact_dir = Path(self.temp_dir) / "artifacts"
        artifact_dir.mkdir()
        test_file = artifact_dir / "test.txt"
        test_file.write_text("test content")
        
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        result = self.manager.deploy_artifacts(deployment_id, str(artifact_dir))
        
        assert result is True
        assert self.manager.deployments[deployment_id].checksum != ""
    
    def test_04_health_check(self):
        """Test health check"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        health_result = self.manager.health_check(deployment_id)
        
        assert health_result.service_name == "rag-core"
        assert isinstance(health_result.status, bool)
    
    def test_05_complete_deployment(self):
        """Test completing deployment"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        result = self.manager.complete_deployment(deployment_id)
        
        assert result is True
        assert self.manager.deployments[deployment_id].status == DeploymentStatus.COMPLETED.value
    
    def test_06_fail_deployment(self):
        """Test failing deployment"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        result = self.manager.fail_deployment(deployment_id, "Test error")
        
        assert result is True
        assert self.manager.deployments[deployment_id].status == DeploymentStatus.FAILED.value
        assert self.manager.deployments[deployment_id].error_message == "Test error"
    
    def test_07_get_deployment_status(self):
        """Test getting deployment status"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        self.manager.complete_deployment(deployment_id)
        
        status = self.manager.get_deployment_status(deployment_id)
        
        assert status["deployment_id"] == deployment_id
        assert status["status"] == DeploymentStatus.COMPLETED.value
        assert status["service_name"] == "rag-core"
    
    def test_08_deployment_history(self):
        """Test deployment history"""
        for i in range(3):
            config = DeploymentConfig(
                service_name="rag-core",
                version=f"2.0.{i}",
                environment=DeploymentEnvironment.STAGING,
                deployment_path="/opt/rag",
            )
            deployment_id = self.manager.start_deployment(config)
            self.manager.complete_deployment(deployment_id)
        
        history = self.manager.get_deployment_history("rag-core")
        assert len(history) >= 3
    
    def test_09_active_deployments(self):
        """Test getting active deployments"""
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.manager.start_deployment(config)
        active = self.manager.get_active_deployments()
        
        assert len(active) >= 1
        assert any(d["deployment_id"] == deployment_id for d in active)
    
    def test_10_deployment_statistics(self):
        """Test deployment statistics"""
        config1 = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        config2 = DeploymentConfig(
            service_name="rag-core",
            version="2.0.1",
            environment=DeploymentEnvironment.STAGING,
            deployment_path="/opt/rag",
        )
        
        dep_id1 = self.manager.start_deployment(config1)
        dep_id2 = self.manager.start_deployment(config2)
        
        self.manager.complete_deployment(dep_id1)
        self.manager.fail_deployment(dep_id2, "Test failure")
        
        stats = self.manager.get_deployment_statistics()
        
        assert stats["total_deployments"] == 2
        assert stats["successful_deployments"] == 1
        assert stats["failed_deployments"] == 1


class TestMigrationSystem:
    """Test Migration System"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.migration_root = f"{self.temp_dir}/migrations"
        self.system = MigrationSystem(migration_root=self.migration_root)
    
    def teardown_method(self):
        """Cleanup test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_11_create_migration(self):
        """Test creating migration"""
        migration_id = self.system.create_migration(
            name="add_user_table",
            script="CREATE TABLE users (id INT PRIMARY KEY);",
            migration_type=MigrationType.SCHEMA,
            description="Add users table",
        )
        
        assert migration_id is not None
        assert migration_id in self.system.migrations
    
    def test_12_validate_migration(self):
        """Test validating migration"""
        migration_id = self.system.create_migration(
            name="add_user_table",
            script="CREATE TABLE users (id INT PRIMARY KEY);",
            migration_type=MigrationType.SCHEMA,
        )
        
        validation = self.system.validate_migration(migration_id)
        
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
    
    def test_13_validate_empty_migration(self):
        """Test validating empty migration"""
        migration_id = self.system.create_migration(
            name="empty_migration",
            script="",
            migration_type=MigrationType.SCHEMA,
        )
        
        validation = self.system.validate_migration(migration_id)
        
        assert validation["valid"] is False
        assert any("empty" in issue.lower() for issue in validation["issues"])
    
    def test_14_execute_migration(self):
        """Test executing migration"""
        migration_id = self.system.create_migration(
            name="add_user_table",
            script="CREATE TABLE users (id INT PRIMARY KEY);",
            migration_type=MigrationType.SCHEMA,
            description="Add users table",
        )
        
        result = self.system.execute_migration(migration_id)
        
        assert result["status"] == "completed"
        assert migration_id in [r.migration_id for r in self.system.migration_history]
    
    def test_15_migration_with_dependencies(self):
        """Test migration with dependencies"""
        dep_migration_id = self.system.create_migration(
            name="add_users_table",
            script="CREATE TABLE users (id INT);",
            migration_type=MigrationType.SCHEMA,
        )
        
        self.system.execute_migration(dep_migration_id)
        
        # Create migration with dependency
        migration_id = self.system.create_migration(
            name="add_posts_table",
            script="CREATE TABLE posts (id INT, user_id INT);",
            migration_type=MigrationType.SCHEMA,
            dependencies=[dep_migration_id],
        )
        
        result = self.system.execute_migration(migration_id)
        assert result["status"] == "completed"
    
    def test_16_rollback_migration(self):
        """Test rolling back migration"""
        migration_id = self.system.create_migration(
            name="add_user_table",
            script="CREATE TABLE users (id INT);",
            migration_type=MigrationType.SCHEMA,
            rollback_script="DROP TABLE users;",
        )
        
        self.system.execute_migration(migration_id)
        result = self.system.rollback_migration(migration_id)
        
        assert result["status"] == "rolled_back"
    
    def test_17_migration_history(self):
        """Test migration history"""
        for i in range(3):
            migration_id = self.system.create_migration(
                name=f"migration_{i}",
                script=f"-- Migration {i}",
                migration_type=MigrationType.DATA,
            )
            self.system.execute_migration(migration_id)
        
        history = self.system.get_migration_history(limit=10)
        assert len(history) >= 3
    
    def test_18_pending_migrations(self):
        """Test getting pending migrations"""
        self.system.create_migration(
            name="migration_1",
            script="-- Migration 1",
            migration_type=MigrationType.SCHEMA,
        )
        self.system.create_migration(
            name="migration_2",
            script="-- Migration 2",
            migration_type=MigrationType.DATA,
        )
        
        pending = self.system.get_pending_migrations()
        assert len(pending) >= 2
    
    def test_19_schema_compatibility(self):
        """Test schema compatibility checking"""
        compat = self.system.check_schema_compatibility("1.1.0")
        
        assert compat["target_version"] == "1.1.0"
        assert "compatible" in compat
        assert "breaking_changes" in compat
    
    def test_20_migration_statistics(self):
        """Test migration statistics"""
        for i in range(2):
            migration_id = self.system.create_migration(
                name=f"migration_{i}",
                script=f"-- Migration {i}",
                migration_type=MigrationType.SCHEMA,
            )
            self.system.execute_migration(migration_id)
        
        stats = self.system.get_migration_statistics()
        
        assert stats["total_migrations"] >= 2
        assert stats["executed_migrations"] >= 2
        assert stats["pending_migrations"] >= 0


class TestRollbackSystem:
    """Test Rollback System"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_root = f"{self.temp_dir}/checkpoints"
        self.system = RollbackSystem(checkpoint_root=self.checkpoint_root)
    
    def teardown_method(self):
        """Cleanup test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_21_create_checkpoint(self):
        """Test creating checkpoint"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
            description="Manual backup",
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in self.system.checkpoints
    
    def test_22_list_checkpoints(self):
        """Test listing checkpoints"""
        self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        self.system.create_checkpoint(
            name="backup_v2",
            version="1.0.0",
            checkpoint_type=CheckpointType.AUTO,
        )
        
        checkpoints = self.system.list_checkpoints(version="1.0.0")
        assert len(checkpoints) >= 2
    
    def test_23_create_recovery_plan(self):
        """Test creating recovery plan"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        plan_id = self.system.create_recovery_plan(
            target_version="1.0.0",
            target_checkpoint_id=checkpoint_id,
        )
        
        assert plan_id is not None
        assert plan_id in self.system.recovery_plans
    
    def test_24_validate_recovery_plan(self):
        """Test validating recovery plan"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        plan_id = self.system.create_recovery_plan(
            target_version="1.0.0",
            target_checkpoint_id=checkpoint_id,
        )
        
        validation = self.system.validate_recovery_plan(plan_id)
        
        assert validation["valid"] is True
    
    def test_25_execute_recovery_plan(self):
        """Test executing recovery plan"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        plan_id = self.system.create_recovery_plan(
            target_version="1.0.0",
            target_checkpoint_id=checkpoint_id,
        )
        
        result = self.system.execute_recovery_plan(plan_id)
        
        assert result["status"] == "completed"
    
    def test_26_manual_rollback(self):
        """Test manual rollback"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        self.system.current_version = "2.0.0"
        
        result = self.system.manual_rollback(
            target_version="1.0.0",
            reason="Emergency rollback",
        )
        
        assert result["status"] == "completed"
    
    def test_27_rollback_history(self):
        """Test rollback history"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        plan_id = self.system.create_recovery_plan(
            target_version="1.0.0",
            target_checkpoint_id=checkpoint_id,
        )
        
        self.system.execute_recovery_plan(plan_id)
        
        history = self.system.get_rollback_history(limit=10)
        assert len(history) >= 1
    
    def test_28_checkpoint_info(self):
        """Test getting checkpoint info"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        info = self.system.get_checkpoint_info(checkpoint_id)
        
        assert info["checkpoint_id"] == checkpoint_id
        assert info["version"] == "1.0.0"
        assert "age_hours" in info
    
    def test_29_checkpoint_with_data(self):
        """Test creating checkpoint with data"""
        # Create test data
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("test")
        
        checkpoint_id = self.system.create_checkpoint(
            name="backup_with_data",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
            data_path=str(data_dir),
        )
        
        info = self.system.get_checkpoint_info(checkpoint_id)
        assert info["size_bytes"] > 0
    
    def test_30_rollback_statistics(self):
        """Test rollback statistics"""
        checkpoint_id = self.system.create_checkpoint(
            name="backup_v1",
            version="1.0.0",
            checkpoint_type=CheckpointType.MANUAL,
        )
        
        plan_id = self.system.create_recovery_plan(
            target_version="1.0.0",
            target_checkpoint_id=checkpoint_id,
        )
        
        self.system.execute_recovery_plan(plan_id)
        
        stats = self.system.get_rollback_statistics()
        
        assert stats["total_checkpoints"] >= 1
        assert stats["successful_rollbacks"] >= 1
        assert "current_version" in stats


class TestIntegratedDeploymentSystem:
    """Integrated tests for deployment system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.deployment_root = f"{self.temp_dir}/deployments"
        self.migration_root = f"{self.temp_dir}/migrations"
        self.checkpoint_root = f"{self.temp_dir}/checkpoints"
        
        self.deployment_manager = DeploymentManager(deployment_root=self.deployment_root)
        self.migration_system = MigrationSystem(migration_root=self.migration_root)
        self.rollback_system = RollbackSystem(checkpoint_root=self.checkpoint_root)
    
    def teardown_method(self):
        """Cleanup test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_31_integrated_deployment_workflow(self):
        """Test complete deployment workflow"""
        # Create checkpoint before deployment
        checkpoint_id = self.rollback_system.create_checkpoint(
            name="pre_deployment",
            version="1.0.0",
            checkpoint_type=CheckpointType.PRE_DEPLOYMENT,
        )
        
        # Create and execute migration
        migration_id = self.migration_system.create_migration(
            name="v2_migration",
            script="-- Schema changes for v2",
            migration_type=MigrationType.SCHEMA,
        )
        
        migration_result = self.migration_system.execute_migration(migration_id)
        assert migration_result["status"] == "completed"
        
        # Deploy new version
        config = DeploymentConfig(
            service_name="rag-core",
            version="2.0.0",
            environment=DeploymentEnvironment.PRODUCTION,
            deployment_path="/opt/rag",
        )
        
        deployment_id = self.deployment_manager.start_deployment(config)
        self.deployment_manager.complete_deployment(deployment_id)
        
        # Verify deployment
        status = self.deployment_manager.get_deployment_status(deployment_id)
        assert status["status"] == DeploymentStatus.COMPLETED.value
    
    def test_32_end_to_end_system_deployment(self):
        """Test end-to-end system deployment"""
        # Create multiple checkpoints
        for i in range(3):
            self.rollback_system.create_checkpoint(
                name=f"checkpoint_{i}",
                version=f"1.{i}.0",
                checkpoint_type=CheckpointType.AUTO,
            )
        
        # Execute multiple migrations
        migration_ids = []
        for i in range(3):
            mid = self.migration_system.create_migration(
                name=f"migration_{i}",
                script=f"-- Migration {i}",
                migration_type=MigrationType.DATA,
            )
            self.migration_system.execute_migration(mid)
            migration_ids.append(mid)
        
        # Deploy multiple versions
        for i in range(3):
            config = DeploymentConfig(
                service_name="rag-core",
                version=f"1.{i}.0",
                environment=DeploymentEnvironment.STAGING,
                deployment_path="/opt/rag",
            )
            dep_id = self.deployment_manager.start_deployment(config)
            self.deployment_manager.complete_deployment(dep_id)
        
        # Verify statistics
        dep_stats = self.deployment_manager.get_deployment_statistics()
        mig_stats = self.migration_system.get_migration_statistics()
        
        assert dep_stats["total_deployments"] >= 3
        assert mig_stats["total_migrations"] >= 3
