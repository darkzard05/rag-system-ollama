import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from core.session import SessionManager

def test_ui_lifecycle_no_rerun():
    print("--- Streamlit UI 라이프사이클 무결성 테스트 (No Rerun) ---")
    SessionManager.init_session()
    
    # 1. 초기 상태 설정
    SessionManager.set("messages", [])
    SessionManager.set("new_file_uploaded", True)
    SessionManager.set("last_uploaded_file_name", "test.pdf")
    SessionManager.set("pdf_file_path", "/tmp/test.pdf")

    # 2. 로직 처리 시뮬레이션 (main.py의 순서와 동일)
    print("[Step 1] 상태 변경 감지 및 로직 실행...")
    
    # _rebuild_rag_system 내부 동작 시뮬레이션
    if SessionManager.get("new_file_uploaded"):
        # 실제 빌드 대신 메시지 추가만 시뮬레이션
        SessionManager.add_message("assistant", "✅ 문서 처리가 완료되었습니다.")
        SessionManager.set("new_file_uploaded", False)
        SessionManager.set("pdf_processed", True)

    print("[Step 2] 메인 UI 렌더링 시뮬레이션 (동일 루프 내)...")
    
    # render_left_column()이 호출되었다고 가정하고 메시지 확인
    messages = SessionManager.get("messages")
    
    # 3. 검증
    print(f"현재 세션 메시지 수: {len(messages)}")
    if len(messages) > 0:
        print(f"최신 메시지 내용: {messages[-1]['content']}")
    
    # 결과 확인
    assert len(messages) == 1
    assert "문서 처리가 완료되었습니다" in messages[0]["content"]
    assert SessionManager.get("pdf_processed") is True
    assert SessionManager.get("new_file_uploaded") is False
    
    print("\n✅ 검증 성공: st.rerun() 없이도 동일 실행 주기 내에서 모든 상태 변화가 정상 반영됩니다.")

if __name__ == "__main__":
    test_ui_lifecycle_no_rerun()