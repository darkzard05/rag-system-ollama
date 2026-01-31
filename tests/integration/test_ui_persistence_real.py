import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session import SessionManager
from core.thread_safe_session import ThreadSafeSessionManager


class TestAppPersistenceFlow(unittest.TestCase):
    def setUp(self):
        """테스트 환경 설정 (Streamlit 환경 모사)"""
        self.mock_session_state = {}
        self.session_id = "test_real_session_123"

        # Streamlit session_state와 컨텍스트 모킹
        self.patcher_state = patch("streamlit.session_state", self.mock_session_state)
        self.patcher_ctx = patch("streamlit.runtime.scriptrunner.get_script_run_ctx")

        self.mock_st_state = self.patcher_state.start()
        self.mock_get_ctx = self.patcher_ctx.start()

        # 컨텍스트 정보 설정
        mock_ctx = MagicMock()
        mock_ctx.session_id = self.session_id
        self.mock_get_ctx.return_value = mock_ctx

        # 초기화
        ThreadSafeSessionManager.clear_session(self.session_id)
        SessionManager.set_session_id(self.session_id)
        SessionManager.init_session()

    def tearDown(self):
        self.patcher_state.stop()
        self.patcher_ctx.stop()

    async def simulate_async_answer(self, user_input: str, answer_text: str):
        """앱의 _stream_chat_response 로직을 시뮬레이션"""
        # 1. 비동기 스레드로 진입 (컨텍스트가 없는 환경 가정)
        with patch(
            "core.thread_safe_session.ThreadSafeSessionManager._is_streamlit_available",
            return_value=False,
        ):
            # 비동기 환경에서 메시지 추가
            SessionManager.add_message("assistant", answer_text)
            SessionManager.add_status_log("답변 완료")

    def test_consecutive_questions_persistence(self):
        """연속 질문 시 데이터 유지 테스트"""

        # --- [STEP 1] 첫 번째 질문 (Main Thread) ---
        q1 = "첫 번째 질문입니다."
        SessionManager.add_message("user", q1)

        # --- [STEP 2] 비동기 답변 생성 시작 (Sync to Fallback) ---
        SessionManager.sync_to_fallback()

        # --- [STEP 3] 비동기 작업 수행 ---
        ans1 = "첫 번째 답변입니다."
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.simulate_async_answer(q1, ans1))

        # --- [STEP 4] 비동기 작업 완료 (Sync from Fallback) ---
        SessionManager.sync_from_fallback()

        # --- [STEP 5] 검증: 첫 번째 대화가 살아있는가? ---
        messages = SessionManager.get_messages()
        assert len(messages) == 2, "첫 번째 질문/답변이 세션에 있어야 합니다."
        assert messages[0]["content"] == q1
        assert messages[1]["content"] == ans1

        # --- [STEP 6] 두 번째 질문 (Main Thread) ---
        q2 = "두 번째 질문입니다."
        SessionManager.add_message("user", q2)

        # --- [STEP 7] 최종 검증: 이전 히스토리가 모두 유지되는가? ---
        final_messages = SessionManager.get_messages()
        assert len(final_messages) == 3, (
            "두 번째 질문 추가 후 총 3개의 메시지가 유지되어야 합니다."
        )
        assert final_messages[0]["content"] == q1
        assert final_messages[1]["content"] == ans1
        assert final_messages[2]["content"] == q2

        print(
            "\n✅ [성공] 비동기 동기화 후에도 이전 답변이 완벽하게 유지됨을 확인했습니다."
        )


if __name__ == "__main__":
    unittest.main()
