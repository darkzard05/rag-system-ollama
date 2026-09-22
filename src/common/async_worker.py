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
