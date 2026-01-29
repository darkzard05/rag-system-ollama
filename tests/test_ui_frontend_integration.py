import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.thread_safe_session import ThreadSafeSessionManager
from core.session import SessionManager


class FrontendIntegrationSimulator(unittest.TestCase):
    def setUp(self):
        """실제 브라우저 세션 환경 모사"""
        self.mock_st_state = {}
        self.session_id = "frontend_test_session_999"

        # 1. 전역 세션 초기화
        ThreadSafeSessionManager.clear_session(self.session_id)
        ThreadSafeSessionManager.set_session_id(self.session_id)

    def simulate_ui_render_cycle(self, user_input=None):
        """Streamlit의 프론트엔드 렌더링 한 주기를 시뮬레이션"""

        # Streamlit 메인 스레드 환경 모킹
        with (
            patch("core.thread_safe_session.st.session_state", self.mock_st_state),
            patch("core.thread_safe_session.get_script_run_ctx") as mock_ctx_mgr,
        ):
            mock_ctx = MagicMock()
            mock_ctx.session_id = self.session_id
            mock_ctx_mgr.return_value = mock_ctx

            # [단계 1] 페이지 로드 시 초기화
            SessionManager.init_session()

            # [단계 2] 현재 메시지 로드 (UI 렌더링 루프)
            messages_before = SessionManager.get_messages()
            print(f"  [UI] 렌더링 중... (메시지 수: {len(messages_before)})")

            # [단계 3] 사용자 입력 발생
            if user_input:
                print(f"  [User] 입력: '{user_input}'")
                SessionManager.add_message("user", user_input)

                # [단계 4] 비동기 답변 생성 시작 (가장 위험한 구간)
                SessionManager.sync_to_fallback()

                # 비동기 태스크 시뮬레이션 (별도 이벤트 루프)
                async def async_task():
                    # ⚠️ Streamlit은 비동기 루프 내에서 st.session_state 접근을 막는 경우가 많음
                    # 이를 위해 _is_streamlit_available을 False로 모킹하여 fallback을 강제함
                    with patch(
                        "core.thread_safe_session.ThreadSafeSessionManager._is_streamlit_available",
                        return_value=False,
                    ):
                        # 비동기 스레드에서 답변 추가
                        ans = f"Ans to {user_input}"
                        SessionManager.add_message("assistant", ans)
                        print("  [Async] 답변 생성 및 저장 완료")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(async_task())

                # [단계 5] 비동기 작업 종료 후 메인 세션으로 복구
                SessionManager.sync_from_fallback()
                print("  [Sync] 데이터를 메인 세션으로 복구함")

    def test_rendering_persistence_across_multiple_runs(self):
        """연속적인 렌더링 주기 동안 데이터가 유지되는지 정밀 테스트"""

        print("\n--- [Run 1] 첫 번째 질문 시나리오 ---")
        self.simulate_ui_render_cycle(user_input="질문 1")

        # Run 1 결과 확인
        self.assertEqual(
            len(self.mock_st_state["messages"]),
            2,
            "첫 사이클 후 메시지는 2개여야 합니다.",
        )

        print("\n--- [Run 2] 리런(Rerun) 또는 단순 페이지 갱신 ---")
        self.simulate_ui_render_cycle()
        self.assertEqual(
            len(self.mock_st_state["messages"]),
            2,
            "리런 후에도 데이터가 유지되어야 합니다.",
        )

        print("\n--- [Run 3] 두 번째 질문 시나리오 ---")
        self.simulate_ui_render_cycle(user_input="질문 2")

        # 최종 상태 확인
        final_history = self.mock_st_state["messages"]
        print("\n[최종 히스토리 점검]")
        for i, m in enumerate(final_history):
            print(f"  {i + 1}. {m['role']}: {m['content']}")

        self.assertEqual(
            len(final_history), 4, "총 4개의 메시지가 완벽하게 보존되어야 합니다."
        )
        self.assertEqual(final_history[0]["content"], "질문 1")
        self.assertEqual(final_history[1]["role"], "assistant")
        self.assertEqual(final_history[2]["content"], "질문 2")
        self.assertEqual(final_history[3]["role"], "assistant")

        print(
            "\n✅ [검증 완료] 프론트엔드 렌더링 주기와 비동기 격리 환경을 모두 통과했습니다."
        )


if __name__ == "__main__":
    unittest.main()
