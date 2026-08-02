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
