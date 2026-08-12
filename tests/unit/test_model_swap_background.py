# INT-3: 모델 교체가 백그라운드 워커에서 실행되는지 검증하는 테스트
# (1) run_in_background_worker 디스패치 (2) is_swapping_model 인플라이트 가드
# (3) asyncio.to_thread로 load_llm 오프로드
import asyncio
import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath("src"))

from core.session import SessionManager
from main import _bg_update_qa_chain, _handle_pending_tasks


class TestModelSwapBackgroundDispatch(unittest.TestCase):
    def setUp(self):
        SessionManager.reset_all_state("test_session")
        SessionManager.set_session_id("test_session")

    def test_qa_chain_update_dispatches_to_background_worker(self):
        """needs_qa_chain_update가 run_in_background_worker로 디스패치되고
        is_swapping_model 인플라이트 플래그가 세팅되는지 검증."""
        SessionManager.set("needs_qa_chain_update", True, session_id="test_session")

        with (
            patch("common.utils.run_in_background_worker") as mock_run_bg,
            patch("main.st.rerun") as mock_rerun,
        ):
            _handle_pending_tasks()

        # 1. 디스패치 검증 — Awaitable(코루틴) + session_id
        mock_run_bg.assert_called_once()
        coro, sid = mock_run_bg.call_args[0]
        assert inspect.isawaitable(coro), "디스패치 대상은 Awaitable 코루틴이어야 함"
        getattr(
            coro, "close", lambda: None
        )()  # mock으로 실행되지 않는 코루틴 자원 해제
        assert sid == "test_session"

        # 2. 가드 플래그 — 스왑 시작 시 세션에 인플라이트 표시
        assert (
            SessionManager.get("is_swapping_model", session_id="test_session") is True
        )
        assert not SessionManager.get(
            "needs_qa_chain_update", session_id="test_session"
        )
        mock_rerun.assert_called_once()

    def test_swap_guard_skips_redispatch_while_swapping(self):
        """스왑 진행 중(is_swapping_model=True) 재변경 요청은 재디스패치되지 않는다."""
        SessionManager.set("needs_qa_chain_update", True, session_id="test_session")
        SessionManager.set("is_swapping_model", True, session_id="test_session")

        with (
            patch("common.utils.run_in_background_worker") as mock_run_bg,
            patch("main.st.rerun"),
        ):
            _handle_pending_tasks()

        # 재디스패치 없음, 요청은 소비되고 인플라이트 플래그는 유지
        mock_run_bg.assert_not_called()
        assert not SessionManager.get(
            "needs_qa_chain_update", session_id="test_session"
        )
        assert (
            SessionManager.get("is_swapping_model", session_id="test_session") is True
        )

    def test_bg_update_qa_chain_offloads_load_llm_via_to_thread(self):
        """_bg_update_qa_chain은 load_llm을 asyncio.to_thread로 오프로드하고
        완료 시 is_swapping_model을 클리어한다."""
        fake_llm = object()
        real_to_thread = asyncio.to_thread
        mock_to_thread = MagicMock(wraps=real_to_thread)
        SessionManager.set(
            "last_selected_model", "test-model", session_id="test_session"
        )

        with (
            patch("core.model_loader.load_llm", return_value=fake_llm) as mock_load,
            patch("asyncio.to_thread", mock_to_thread),
        ):
            asyncio.run(_bg_update_qa_chain("test_session"))

        # to_thread 오프로드 — 단일 AsyncWorker 루프 블로킹 방지
        mock_load.assert_called_once_with("test-model")
        mock_to_thread.assert_called_once_with(mock_load, "test-model")

        # 결과 반영 + 플래그 클리어 + 상태 로그
        assert SessionManager.get("llm", session_id="test_session") is fake_llm
        assert (
            SessionManager.get("is_swapping_model", session_id="test_session") is False
        )
        logs = SessionManager.get("status_logs", [], session_id="test_session") or []
        assert any("교체 완료" in log for log in logs)


if __name__ == "__main__":
    unittest.main()
