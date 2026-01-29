import shutil
import pytest
from infra.deployment_manager import (
    DeploymentManager,
    DeploymentConfig,
    DeploymentEnvironment,
)
from infra.rollback_system import RollbackSystem, CheckpointType


@pytest.fixture
def temp_infra_dir(tmp_path):
    """테스트를 위한 임시 인프라 디렉토리 제공"""
    deploy_root = tmp_path / "deployments"
    checkpoint_root = tmp_path / "checkpoints"
    return deploy_root, checkpoint_root


def test_deployment_failure_cleanup(temp_infra_dir, tmp_path):
    deploy_root, _ = temp_infra_dir
    manager = DeploymentManager(deployment_root=str(deploy_root))

    config = DeploymentConfig(
        service_name="test_leak_service",
        version="1.0.0",
        environment=DeploymentEnvironment.STAGING,
        deployment_path=str(deploy_root),
    )

    dep_id = manager.start_deployment(config)

    # 1. 존재하지 않는 소스 경로로 배포 시도 (실패 유발)
    print(f"[*] Simulating failed deployment for {dep_id}...")
    with pytest.raises(FileNotFoundError):
        manager.deploy_artifacts(dep_id, "non_existent_source_path_12345")

    # 2. 검증: artifacts 디렉토리가 남아있지 않아야 함 (Disk Leak 방지)
    artifacts_dir = deploy_root / dep_id / "artifacts"
    assert not artifacts_dir.exists(), (
        "Failure: Artifacts directory was not cleaned up after error!"
    )
    print("✅ Success: Partial artifacts cleaned up automatically.")


def test_rollback_system_sync(temp_infra_dir, tmp_path):
    _, checkpoint_root = temp_infra_dir
    system = RollbackSystem(checkpoint_root=str(checkpoint_root))

    # 1. 체크포인트 생성
    dummy_data = tmp_path / "dummy.txt"
    dummy_data.write_text("hello")
    cid = system.create_checkpoint(
        "test_sync", "1.0.0", CheckpointType.AUTO, data_path=str(dummy_data)
    )

    # 2. 물리적 디렉토리 강제 삭제 (상태 불일치 유발)
    checkpoint_dir = checkpoint_root / cid
    shutil.rmtree(checkpoint_dir)
    assert cid in system.checkpoints, "Checkpoint should still be in registry"

    # 3. 싱크 실행
    print("[*] Running sync_checkpoints...")
    stats = system.sync_checkpoints()

    # 4. 검증: 레지스트리에서 제거되어야 함
    assert cid not in system.checkpoints, (
        "Failure: Missing checkpoint was not removed from registry during sync!"
    )
    assert stats["removed_records"] == 1
    print("✅ Success: Registry synced with physical file system.")


if __name__ == "__main__":
    # 이 스크립트는 pytest로 실행하는 것을 권장합니다.
    pass
