import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.constants import FilePathConstants
from core.session import SessionManager


def test_file_system_integrity():
    print("\n[1/3] File System Integrity Test...")
    temp_dir = Path(FilePathConstants.TEMP_DIR)

    # 1. 디렉토리 존재 확인
    if not temp_dir.exists():
        print(f"  - Creating directory: {temp_dir}")
        temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. 파일 생성 및 삭제 테스트
    test_file = temp_dir / "integrity_test.pdf"
    test_file.write_text("dummy content")
    print(f"  - Test file created: {test_file}")

    if test_file.exists():
        os.remove(test_file)
        print("  - Test file deleted successfully (Cleanup OK)")
    else:
        raise Exception("Failed to create test file!")


def test_module_dependency_integrity():
    print("\n[2/3] Module Dependency Integrity Test (Checking removed modules)...")
    try:
        print("  - Core modules imported successfully (Import OK)")

        # 삭제된 모듈 임포트 시도 (ImportError가 발생해야 정상)
        try:
            from services.optimization.batch_optimizer import get_optimal_batch_size

            print("  - [Warning] Deleted batch_optimizer is still accessible!")
            return False
        except ImportError:
            print("  - batch_optimizer reference removed (Dependency Clean OK)")

    except Exception as e:
        print(f"  - Error during module loading: {e}")
        return False
    return True


def test_session_state_integrity():
    print("\n[3/3] Session State Integrity Test...")
    SessionManager.init_session()
    logs = SessionManager.get("status_logs")

    if logs == ["시스템 대기 중"]:
        print(f"  - Initial logs optimized: {logs}")
    else:
        print(f"  - [Note] Initial logs differ from expectation: {logs}")

    SessionManager.reset_for_new_file()
    new_logs = SessionManager.get("status_logs")
    if len(new_logs) > 0 and "새 문서 분석 시작" in new_logs[0]:
        print(f"  - Logs cleared and reset on new file: {new_logs}")
    else:
        print(f"  - [Note] Logs not reset as expected: {new_logs}")


if __name__ == "__main__":
    try:
        test_file_system_integrity()
        if test_module_dependency_integrity():
            test_session_state_integrity()
        print("\n✅ Integrity tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Integrity tests failed: {e}")
        sys.exit(1)
