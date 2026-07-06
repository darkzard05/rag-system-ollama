import os
from src.core.thread_safe_session import ThreadSafeSessionManager


def test_physical_file_deletion():
    # 1. 테스트용 임시 파일 생성
    test_file = "test_leak_verify.pdf"
    with open(test_file, "wb") as f:
        f.write(b"%PDF-1.4 test content")

    session_id = "test_session_123"

    # 2. 세션 생성 및 파일 경로 등록
    ThreadSafeSessionManager.set_session_id(session_id)
    ThreadSafeSessionManager.set("pdf_file_path", os.path.abspath(test_file))

    print(f"파일 생성 확인: {os.path.exists(test_file)}")

    # 3. 세션 삭제 호출
    success = ThreadSafeSessionManager.delete_session(session_id)
    print(f"세션 삭제 성공 여부: {success}")

    # 4. 물리 파일 삭제 여부 확인
    exists = os.path.exists(test_file)
    print(f"파일 삭제 후 존재 여부: {exists}")

    if not exists:
        print("✅ 검증 성공: 물리 파일이 정상적으로 삭제되었습니다.")
    else:
        print("❌ 검증 실패: 물리 파일이 여전히 존재합니다.")
        # Cleanup if failed
        if exists:
            os.remove(test_file)


if __name__ == "__main__":
    test_physical_file_deletion()
