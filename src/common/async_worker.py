"""
Thread-safe async worker backed by a dedicated background event loop.

Replaces ``nest_asyncio`` by providing a proper cross-thread coroutine
submission mechanism via ``run_coroutine_threadsafe``.

Usage::

    from common.async_worker import AsyncWorker

    worker = AsyncWorker()                          # singleton
    future = worker.submit(some_coroutine())        # non-blocking
    result = worker.run_sync(some_coroutine())      # blocking
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger(__name__)


class AsyncWorker:
    """Singleton async worker with a dedicated background event loop.

    All coroutines submitted via :meth:`submit` run on the same event loop,
    ensuring thread safety and proper async lifecycle management.
    One instance per process — ``run_coroutine_threadsafe`` makes every
    submission thread-safe regardless of the caller's thread.
    """

    _instance: AsyncWorker | None = None
    _init_lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> AsyncWorker:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AsyncWorkerLoop",
        )
        self._thread.start()
        logger.info("[ASYNC_WORKER] Dedicated event loop started")

    # -- internal -----------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # -- public API ---------------------------------------------------------

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the underlying event loop (read-only for callers)."""
        return self._loop

    def submit(self, coro: Any) -> Future[Any]:
        """Submit a coroutine to the worker's event loop (non-blocking).

        Returns a :class:`~concurrent.futures.Future` that resolves when the
        coroutine completes.  Safe to call from **any** thread.
        """
        if not self._loop.is_running():
            raise RuntimeError("AsyncWorker event loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro: Any) -> Any:
        """Submit a coroutine and **block** until it completes.

        Useful when the caller is on a synchronous thread (e.g. Streamlit's
        main thread) and needs the result before proceeding.
        """
        future = self.submit(coro)
        return future.result()

    def shutdown(self) -> None:
        """Gracefully stop the event loop and join the background thread."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        logger.info("[ASYNC_WORKER] Event loop stopped")


def request_session_rerun(session_id: str) -> bool:
    """Request a Streamlit rerun for the given session id (thread-safe).

    Usable from background worker threads to propagate state changes
    (e.g. build progress) to the frontend. Returns True on success.
    """
    if not session_id:
        return False
    try:
        from streamlit.runtime import get_instance

        runtime = get_instance()
        if runtime is None:
            return False
        session_info = runtime._session_mgr.get_session_info(session_id)
        if session_info is None:
            return False
        session_info.session.request_rerun(None)
        return True
    except Exception as e:  # noqa: BLE001 - rerun request must never crash worker
        logger.debug(f"Session rerun request failed ({session_id}): {e}")
        return False


def run_in_background_worker(coro: Awaitable[Any], session_id: str) -> None:
    """
    Streamlit 환경에서 코루틴을 AsyncWorker의 전용 이벤트 루프에서 실행하는 백그라운드 워커.
    - run_coroutine_threadsafe로 스레드 안전하게 코루틴 제출
    - 작업 완료 후 자동으로 rerun 트리거
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from core.session import SessionManager

    ctx = get_script_run_ctx()

    async def _with_session() -> Any:
        SessionManager.set_session_id(session_id)
        return await coro

    def _on_complete(future):
        worker_error: Exception | None = None
        try:
            future.result()
        except Exception as e:
            worker_error = e
            logger.error(f"Background worker error: {e}", exc_info=True)

        if ctx and ctx.session_id:
            try:
                if not request_session_rerun(ctx.session_id):
                    raise RuntimeError("session not found for rerun")
            except Exception as e:
                # rerun 재요청 실패 시 입력창이 영구 비활성화되지 않도록
                # 세션 플래그를 정리한다(INT-입력동결). D4: 조용한 정상 위장을
                # 막기 위해 실패를 상태 로그에 남겨 사용자에게 표면화한다.
                # 빌드 진행은 워커의 상태 전파 rerun(쓰로틀)으로 갱신되므로,
                # 여기서 플래그를 내려주면 다음 렌더에서 정상 상태로 복구된다.
                logger.error(f"Background worker rerun failed: {e}", exc_info=True)
                try:
                    from core.session import SessionManager

                    SessionManager.set_session_id(ctx.session_id)
                    SessionManager.set("is_building_rag", False, ctx.session_id)
                    SessionManager.set("is_swapping_model", False, ctx.session_id)
                    SessionManager.set("is_generating_answer", False, ctx.session_id)
                    detail = f"{type(e).__name__}: {e}"[:200]
                    SessionManager.add_status_log(
                        "백그라운드 작업 완료 후 화면 갱신에 실패했습니다. "
                        f"페이지를 새로고침해 주세요. ({detail})",
                        session_id=ctx.session_id,
                    )
                except Exception as inner:  # noqa: BLE001 - 복구 실패는 로그만
                    logger.error(f"Flag recovery failed: {inner}", exc_info=True)
        if worker_error is not None and ctx and ctx.session_id:
            # D4: 워커 본체 실패도 로그에만 두지 않고 상태 로그로 표면화한다.
            try:
                from core.session import SessionManager

                SessionManager.set_session_id(ctx.session_id)
                SessionManager.add_status_log(
                    "백그라운드 작업 중 오류가 발생했습니다: "
                    f"{str(worker_error)[:200]}",
                    session_id=ctx.session_id,
                )
            except Exception as inner:  # noqa: BLE001 - 표면화 실패는 로그만
                logger.error(f"Error surfacing failed: {inner}", exc_info=True)

    future = AsyncWorker().submit(_with_session())
    future.add_done_callback(_on_complete)
