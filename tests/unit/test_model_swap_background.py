# INT-3: 모델 교체가 백그라운드 워커에서 실행되는지 검증하는 테스트
# (1) run_in_background_worker 디스패치 (2) is_swapping_model 인플라이트 가드
# (3) asyncio.to_thread로 LLM 스왑 오프로드 (공용 _swap_llm_core 위임, R: Group 10)
import asyncio
import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath("src"))

from src.main import _bg_update_qa_chain, _handle_pending_tasks, _swap_llm_core

from core.session import SessionManager


class TestModelSwapBackgroundDispatch(unittest.TestCase):
    def setUp(self):
        SessionManager.reset()
        SessionManager.reset_all_state("test_session")
        SessionManager.set_ui_sync(None)
        SessionManager.set_session_id("test_session")

    def test_qa_chain_update_dispatches_to_background_worker(self):
        """needs_qa_chain_update가 run_in_background_worker로 디스패치되고
        is_swapping_model 인플라이트 플래그가 세팅되는지 검증."""
        SessionManager.set("needs_qa_chain_update", True, session_id="test_session")

        with (
            patch("common.utils.run_in_background_worker") as mock_run_bg,
            patch("src.main.st.rerun") as mock_rerun,
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
            patch("src.main.st.rerun"),
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

    def test_bg_update_qa_chain_offloads_core_via_to_thread(self):
        """_bg_update_qa_chain은 공용 _swap_llm_core를 asyncio.to_thread로
        오프로드하고 완료 시 is_swapping_model을 클리어한다."""
        SessionManager.set(
            "last_selected_model", "test-model", session_id="test_session"
        )
        fake_llm = object()

        # to_thread를 실제 코어 실행으로 대체 (오프로드 대상/인자 검증용)
        async def _run_core(func, arg):
            func(arg)

        mock_to_thread = MagicMock(wraps=_run_core)

        with (
            patch("core.model_loader.load_llm", return_value=fake_llm) as mock_load,
            patch("asyncio.to_thread", mock_to_thread),
        ):
            asyncio.run(_bg_update_qa_chain("test_session"))

        # to_thread 오프로드 — 단일 AsyncWorker 루프 블로킹 방지
        mock_to_thread.assert_called_once_with(_swap_llm_core, "test_session")
        # 오프로드된 코어가 실제로 모델을 로드했다
        mock_load.assert_called_once_with("test-model")
        # 완료 시 인플라이트 플래그 클리어
        assert (
            SessionManager.get("is_swapping_model", session_id="test_session") is False
        )

    def test_swap_llm_core_loads_model_and_sets_llm(self):
        """공용 _swap_llm_core는 모델을 로드해 세션에 LLM을 세팅하고 완료 플래그를 남긴다."""
        fake_llm = object()
        SessionManager.set(
            "last_selected_model", "test-model", session_id="test_session"
        )

        with patch("core.model_loader.load_llm", return_value=fake_llm) as mock_load:
            _swap_llm_core("test_session")

        mock_load.assert_called_once_with("test-model")
        assert SessionManager.get("llm", session_id="test_session") is fake_llm
        assert (
            SessionManager.get("rag_build_complete_flag", session_id="test_session")
            is True
        )


if __name__ == "__main__":
    unittest.main()
