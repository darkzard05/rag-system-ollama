"""Domain utilities extracted from ``common.utils`` (Batch 5.4).
Log namespace preserved as ``common.utils`` for compatibility.
"""

import logging
from collections.abc import Awaitable
from typing import Any

logger = logging.getLogger("common.utils")


def run_in_background_worker(coro: Awaitable[Any], session_id: str) -> None:
    """
    Streamlit 환경에서 코루틴을 AsyncWorker의 전용 이벤트 루프에서 실행하는 백그라운드 워커.
    - run_coroutine_threadsafe로 스레드 안전하게 코루틴 제출
    - 작업 완료 후 자동으로 rerun 트리거
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from common.async_worker import AsyncWorker
    from core.session import SessionManager

    ctx = get_script_run_ctx()

    async def _with_session() -> Any:
        SessionManager.set_session_id(session_id)
        return await coro

    def _on_complete(future):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Background worker error: {e}", exc_info=True)

        if ctx and ctx.session_id:
            try:
                from streamlit.runtime import get_instance

                runtime = get_instance()
                if runtime:
                    session_info = runtime._session_mgr.get_session_info(ctx.session_id)
                    if session_info:
                        session_info.session.request_rerun(None)
            except Exception as e:
                # rerun 재요청 실패 시 입력창이 영구 비활성화되지 않도록
                # 세션 플래그를 정리한다(INT-입력동결). 빌드 진행 바는
                # 전용 폴링 fragment(_render_build_progress_fragment, 1.5초)가
                # 플래그를 읽어 갱신하므로, 여기서 플래그를 내려주면
                # 다음 폴링에서 정상 상태로 복구된다.
                logger.error(f"Background worker rerun failed: {e}", exc_info=True)
                try:
                    from core.session import SessionManager

                    SessionManager.set_session_id(ctx.session_id)
                    SessionManager.set("is_building_rag", False, ctx.session_id)
                    SessionManager.set("is_swapping_model", False, ctx.session_id)
                    SessionManager.set("is_generating_answer", False, ctx.session_id)
                except Exception as inner:  # noqa: BLE001 - 복구 실패는 로그만
                    logger.error(f"Flag recovery failed: {inner}", exc_info=True)

    future = AsyncWorker().submit(_with_session())
    future.add_done_callback(_on_complete)
