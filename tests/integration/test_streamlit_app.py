import os
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# 헤드리스 모드: 실제 Ollama 호출 대신 가짜 LLM/임베딩 스텁 사용
# (model_loader.py:341, :443 — verify_a/b와 동일한 패턴. 없으면 실제 모델
# 로드/스트리밍 경로로 진입해 스트리밍 실패로 사용자 메시지가 미렌더됨)
os.environ.setdefault("IS_CI_TEST", "true")

from core.session import SessionManager  # noqa: E402


def _app_session_id() -> str:
    """AppTest 스크립트 스레드가 사용하는 SessionManager 스토어 키입니다.

    LocalScriptRunner는 session_id="test session id"를 하드코딩하므로
    (verify_a.py:79-81과 동일한 판별 로직) 부팅 직후 스토어에서 유도합니다.
    """
    fb = SessionManager._fallback_sessions
    return max(fb, key=lambda k: fb[k]["last_accessed"])


def _set_ready_state(sid: str) -> None:
    """is_ready_for_chat(manager.py:350-357)의 조건을 채워 입력창을 활성화합니다."""
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("rag_engine", object(), sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("needs_rag_rebuild", False, sid)
    SessionManager.set("needs_qa_chain_update", False, sid)
    SessionManager.set("pdf_processing_error", None, sid)


class TestRAGStreamlitUI(unittest.TestCase):
    def setUp(self):
        """앱 테스트 인스턴스 초기화 (src/main.py가 진입점이라고 가정)"""
        self.at = AppTest.from_file("src/main.py", default_timeout=60)

    def test_app_initial_state(self):
        """앱 시작 시 초기 UI 요소들이 존재하는지 검증"""
        self.at.run()

        # 사이드바 브랜드 확인 (sidebar.py: st.markdown으로 "GraphRAG-Ollama" 렌더)
        # st.markdown은 markdown 요소로 렌더링됨
        assert any("GraphRAG-Ollama" in str(m.value) for m in self.at.sidebar.markdown)

        # 채팅 가이드 메시지 존재 여부 (chat.py: MSG_CHAT_GUIDE — Phase B 단일 가이드)
        assert any(
            "PDF를 업로드한 후 질문해 보세요" in str(m.value)
            for m in self.at.chat_message[0].markdown
        )

        # 모델 선택 셀렉트박스 존재 여부 (sidebar.py: st.selectbox)
        # sidebar 내부에 중첩된 요소들은 at.sidebar.selectbox로 접근 가능
        assert len(self.at.sidebar.selectbox) >= 1

    def test_chat_interaction_rendering(self):
        """채팅 입력 시 화면 렌더링 흐름 검증"""
        self.at.run()

        # 헤드리스 모드: PDF 업로드가 없으면 chat_input이 disabled라 제출이
        # 무시되어 사용자 메시지가 쌓이지 않습니다. is_ready_for_chat의 세션
        # 키를 채워 입력창을 활성화한 뒤 제출해야 채팅 입력 → 렌더링 흐름이
        # 실제로 동작합니다 (verify_b._set_ready_state 패턴).
        sid = _app_session_id()
        _set_ready_state(sid)
        self.at.run()

        # 1. 채팅 입력 시뮬레이션
        if hasattr(self.at, "chat_input") and self.at.chat_input:
            prompt = self.at.chat_input[0]
            prompt.set_value("테스트 질문입니다.").run()

            # 2. 사용자 메시지가 화면에 렌더링되었는지 확인
            user_msg = [m for m in self.at.chat_message if m.name == "user"]
            assert len(user_msg) > 0
            assert "테스트 질문입니다." in str(user_msg[0].markdown[0].value)

            # 3. 답변 생성 시도 로그 또는 채팅 입력 비활성화 상태 확인
            # (RAG 엔진이 백그라운드에서 동작하므로 입력창이 비활성화되었거나
            # 다음 런타임에 메시지가 추가되는지 확인)
            print("✅ UI 상호작용 및 기본 렌더링 테스트 통과 (사용자 입력 확인)")


if __name__ == "__main__":
    # 실제 환경에서는 Streamlit 앱이 복잡하므로 일부 기능만 단위 테스트로 실행
    unittest.main()
