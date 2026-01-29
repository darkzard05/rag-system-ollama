import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.thread_safe_session import ThreadSafeSessionManager
from core.session import SessionManager


class HighFidelityAppSimulator(unittest.TestCase):
    def setUp(self):
        """실제 브라우저 세션과 같은 영속적 환경 설정"""
        # SessionManager가 사용할 실제 딕셔너리
        self.shared_storage = {}
        self.session_id = "browser_session_001"

        # 저장소 초기화
        ThreadSafeSessionManager.clear_session(self.session_id)
        ThreadSafeSessionManager.set_session_id(self.session_id)

    def run_streamlit_cycle(self, user_input: str = None):
        """Streamlit 단일 사이클 시뮬레이션"""

        # 1. ThreadSafeSessionManager가 참조하는 전역 st 모듈들을 모두 패치
        # 이 부분이 핵심입니다. 여러 곳에서 st를 임포트하므로 일관된 패치가 필요합니다.
        with (
            patch("core.thread_safe_session.st.session_state", self.shared_storage),
            patch("core.thread_safe_session.get_script_run_ctx") as mock_ctx_mgr,
            patch("streamlit.session_state", self.shared_storage),
        ):
            # 컨텍스트 설정
            mock_ctx = MagicMock()
            mock_ctx.session_id = self.session_id
            mock_ctx_mgr.return_value = mock_ctx

            # [시작] 앱 초기화
            SessionManager.init_session()

            # [렌더링]
            messages = SessionManager.get_messages()
            print(f"\n[Cycle Render] Messages in UI: {len(messages)}")

            # [입력 처리]
            if user_input:
                print(f"[User Input] '{user_input}'")
                SessionManager.add_message("user", user_input)

                # 비동기 작업 전 동기화
                SessionManager.sync_to_fallback()

                async def generate_response():
                    # 비동기 환경에서는 Streamlit 접근 불가 강제
                    with patch(
                        "core.thread_safe_session.ThreadSafeSessionManager._is_streamlit_available",
                        return_value=False,
                    ):
                        ans = f"Ans for {user_input} [p.1]"
                        SessionManager.add_message("assistant", ans)

                loop = asyncio.get_event_loop()
                loop.run_until_complete(generate_response())

                # 비동기 작업 후 동기화
                SessionManager.sync_from_fallback()
                print("[Sync Done] Fallback -> Main State")

    def test_high_fidelity_persistence(self):
        """전체 라이프사이클 검증"""
        print("\n--- Start Simulation ---")

        # 1. 첫 번째 질문
        self.run_streamlit_cycle("Q1: Hello?")
        self.assertEqual(len(self.shared_storage.get("messages", [])), 2)

        # 2. Rerun (입력 없음)
        self.run_streamlit_cycle()
        self.assertEqual(
            len(self.shared_storage.get("messages", [])),
            2,
            "Rerun 후에도 데이터가 유지되어야 함",
        )

        # 3. 두 번째 질문
        self.run_streamlit_cycle("Q2: How are you?")

        # 최종 결과
        final_msgs = self.shared_storage.get("messages", [])
        print(f"\n[Final Results] Total count: {len(final_msgs)}")
        for m in final_msgs:
            print(f"  - {m['role']}: {m['content']}")

        self.assertEqual(len(final_msgs), 4)
        self.assertEqual(final_msgs[0]["content"], "Q1: Hello?")
        self.assertEqual(final_msgs[1]["role"], "assistant")
        self.assertEqual(final_msgs[2]["content"], "Q2: How are you?")
        self.assertEqual(final_msgs[3]["role"], "assistant")

        print(
            "\n✅ [성공] 고충실도 시뮬레이션 결과, 모든 데이터가 완벽하게 유지됩니다."
        )


if __name__ == "__main__":
    unittest.main()
