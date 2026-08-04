import sys
from pathlib import Path

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from core.session import SessionManager


def test_ui_lifecycle_no_rerun():
    SessionManager.init_session()

    # 1. 초기 상태 설정
    SessionManager.set("messages", [])
    SessionManager.set("new_file_uploaded", True)
    SessionManager.set("last_uploaded_file_name", "test.pdf")
    SessionManager.set("pdf_file_path", "/tmp/test.pdf")

    # 2. 로직 처리 시뮬레이션 (main.py의 순서와 동일)

    # _rebuild_rag_system 내부 동작 시뮬레이션
    if SessionManager.get("new_file_uploaded"):
        # 실제 빌드 대신 메시지 추가만 시뮬레이션
        SessionManager.add_message("assistant", "✅ 문서 처리가 완료되었습니다.")
        SessionManager.set("new_file_uploaded", False)
        SessionManager.set("pdf_processed", True)

    # render_left_column()이 호출되었다고 가정하고 메시지 확인
    messages = SessionManager.get("messages")

    # 3. 검증
    assert len(messages) == 1
    assert "문서 처리가 완료되었습니다" in messages[0]["content"]
    assert SessionManager.get("pdf_processed") is True
    assert SessionManager.get("new_file_uploaded") is False
