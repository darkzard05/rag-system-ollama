"""Bounded UI stream-slot gating and cancellable stream-task registry.

The UI can run multiple streaming generations concurrently, but each one
consumes a bounded resource (Ollama VRAM / host RAM). This module provides a
global semaphore that caps the number of concurrently ACTIVE stream slots, plus
a small registry of running stream tasks so a run can be cancelled by id.

The semaphore is built lazily by ``_stream_semaphore()`` and NEVER at import
time: reading the limit through the config module object at call time
(``common.config.UI_STREAM_WORKERS``) means an environment override applied
after import still takes effect. This mirrors the read-at-call-time philosophy
of ``config.is_test_env()``. The semaphore is only rebuilt while no slot is
active, so a mid-flight limit change defers the swap to the next idle
transition and never breaks the in-place bound.
"""

import asyncio
import logging
import threading
import time

import common.config as _config

logger = logging.getLogger(__name__)

_sem_lock = threading.Lock()
_stream_sem: threading.Semaphore | None = None
_last_limit: int = 0

_stream_lock = threading.Lock()
_active_streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task[None]]] = {}
_active_slots: int = 0
_waiting: int = 0


def _stream_semaphore() -> threading.Semaphore:
    """Return the current stream semaphore, rebuilding it on idle limit changes."""
    global _stream_sem, _last_limit
    limit = int(_config.UI_STREAM_WORKERS)
    with _sem_lock:
        with _stream_lock:
            active = _active_slots
        if _stream_sem is None or (limit != _last_limit and active == 0):
            _stream_sem = threading.Semaphore(limit)
            _last_limit = limit
        return _stream_sem


def active_stream_count() -> int:
    """Number of slots currently acquired (concurrently active streams)."""
    with _stream_lock:
        return _active_slots


def waiting_stream_count() -> int:
    """Number of threads blocked on ``acquire_stream_slot`` right now."""
    with _stream_lock:
        return _waiting


def acquire_stream_slot(timeout: float) -> bool:
    """Block up to ``timeout`` seconds for a free stream slot; True on success."""
    global _active_slots, _waiting
    sem = _stream_semaphore()
    with _stream_lock:
        _waiting += 1
    ok = sem.acquire(timeout=timeout)
    with _stream_lock:
        _waiting -= 1
        if ok:
            _active_slots += 1
    return ok


def release_stream_slot() -> None:
    """Release a previously acquired stream slot."""
    global _active_slots
    with _stream_lock:
        _active_slots -= 1
    _stream_semaphore().release()


def register_stream(
    run_id: str, loop: asyncio.AbstractEventLoop, task: asyncio.Task[None]
) -> None:
    """Remember the task handle for ``run_id`` so it can be cancelled later."""
    with _stream_lock:
        _active_streams[run_id] = (loop, task)


def unregister_stream(run_id: str) -> None:
    """Drop the task handle for ``run_id`` (no-op when it was never registered)."""
    with _stream_lock:
        _active_streams.pop(run_id, None)


def cancel_stream(run_id: str) -> None:
    """Cancel the task registered under ``run_id``, then briefly drain its loop."""
    with _stream_lock:
        handle = _active_streams.get(run_id)
    if handle is None:
        logger.warning("cancel_stream: no active stream for %s", run_id)
        return
    loop, task = handle
    try:
        loop.call_soon_threadsafe(task.cancel)
    except RuntimeError:
        logger.warning("cancel_stream: event loop for %s is closed", run_id)
        return
    _ensure_cancel_processed(loop, 0.05)


def _ensure_cancel_processed(loop: asyncio.AbstractEventLoop, timeout: float) -> None:
    """Give the loop a brief opportunity to actually process the cancel callback.

    Thread-safety: never drive the loop from a non-owning thread. When the loop
    is running, schedule a no-op callback and yield ``timeout`` so its owner
    processes the pending cancel. When it is NOT running, do nothing: the
    loop's driving thread either already finished the task (nothing to cancel)
    or is about to close the loop — calling ``run_until_complete``/``close``
    here would race that close (Windows Proactor: ``_ssock`` becomes None and
    ``loop.close()`` raises AttributeError / Python assert).
    """

    if not loop.is_running():
        return

    async def _pause() -> None:
        await asyncio.sleep(0)

    loop.call_soon_threadsafe(loop.create_task, _pause())
    time.sleep(min(timeout, 0.05))
