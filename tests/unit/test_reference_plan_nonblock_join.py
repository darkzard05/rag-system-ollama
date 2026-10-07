"""Task 1 non-blocking prewarm join (TDD failing-first).

Render thread must never block on the prewarm future: ``adopt`` returns
immediately (no rects) when the future is pending, keeps the future
registered, and hydrates ``SessionManager.pdf_annotations`` via a done
callback that the viewer click path picks up. Proves ``[PERF] join-block``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from typing import Any

from core.session import SessionManager
from ui.components import reference_plan as rp_mod


def _seed_plan(
    sid: str, msg_id: str, docs: list[Any], cits: list[dict[str, Any]]
) -> Any:
    plan = rp_mod._build_reference_plan(docs, cits, msg_id)
    with rp_mod._registry_lock:
        rp_mod._ACTIVE_PLANS[(sid, msg_id)] = plan
    return plan


def _cleanup(
    sid: str, msg_id: str, sig: str, fut: Future[list[dict[str, Any]]]
) -> None:
    with rp_mod._registry_lock:
        rp_mod._ACTIVE_PLANS.pop((sid, msg_id), None)
        rp_mod._PREWARM_FUTURES.pop(sig, None)
    if not fut.done():
        fut.cancel()
    SessionManager.delete("pdf_annotations", sid)
    SessionManager.delete("file_hash", sid)


def test_adopt_pending_future_returns_fast_without_rects(caplog: Any) -> None:
    """RED: slow future must not stall the render thread (>50ms = fail)."""
    sid = "task1-nonblock-pending"
    msg_id = "m-pending"
    docs: list[Any] = [{"page_content": "hello world", "metadata": {"page": 1}}]
    cits: list[dict[str, Any]] = []
    plan = _seed_plan(sid, msg_id, docs, cits)
    fut: Future[list[dict[str, Any]]] = Future()
    with rp_mod._registry_lock:
        rp_mod._PREWARM_FUTURES[plan.docs_signature] = fut
    try:
        with caplog.at_level(logging.INFO):
            t0 = time.perf_counter()
            got = rp_mod.adopt_prewarmed_annotations(
                sid, msg_id, docs, cits, wait_s=2.0
            )
            elapsed = time.perf_counter() - t0
        assert got is None, "pending prewarm renders without rects"
        assert elapsed < 0.05, f"render thread blocked {elapsed:.3f}s on prewarm"
        assert "[PERF] join-block" in caplog.text
        with rp_mod._registry_lock:
            assert rp_mod._PREWARM_FUTURES.get(plan.docs_signature) is fut
    finally:
        _cleanup(sid, msg_id, plan.docs_signature, fut)


def test_pending_callback_hydrates_session_on_completion() -> None:
    """Done callback writes pdf_annotations; viewer click path reads it."""
    sid = "task1-nonblock-callback"
    msg_id = "m-callback"
    docs: list[Any] = [{"page_content": "hello world", "metadata": {"page": 1}}]
    cits: list[dict[str, Any]] = []
    plan = _seed_plan(sid, msg_id, docs, cits)
    SessionManager.set("file_hash", "hash-cb", session_id=sid)
    fut: Future[list[dict[str, Any]]] = Future()
    rects = [{"page": 1, "x": 1.0, "y": 2.0}]
    with rp_mod._registry_lock:
        rp_mod._PREWARM_FUTURES[plan.docs_signature] = fut
    try:
        got = rp_mod.adopt_prewarmed_annotations(sid, msg_id, docs, cits)
        assert got is None
        fut.set_result(rects)  # fires done callbacks synchronously
        stored = SessionManager.get("pdf_annotations", None, sid)
        assert isinstance(stored, dict)
        assert stored["annotations"] == rects
        assert stored["file_hash"] == "hash-cb"
        with rp_mod._registry_lock:
            assert plan.docs_signature not in rp_mod._PREWARM_FUTURES
    finally:
        _cleanup(sid, msg_id, plan.docs_signature, fut)


def test_done_future_returns_rects_and_pops_registry(caplog: Any) -> None:
    """Already-warm future still adopts synchronously with [PERF] span."""
    sid = "task1-nonblock-done"
    msg_id = "m-done"
    docs: list[Any] = [{"page_content": "hello world", "metadata": {"page": 1}}]
    cits: list[dict[str, Any]] = []
    plan = _seed_plan(sid, msg_id, docs, cits)
    fut: Future[list[dict[str, Any]]] = Future()
    rects = [{"page": 1, "x": 3.0, "y": 4.0}]
    fut.set_result(rects)
    with rp_mod._registry_lock:
        rp_mod._PREWARM_FUTURES[plan.docs_signature] = fut
    try:
        with caplog.at_level(logging.INFO):
            t0 = time.perf_counter()
            got = rp_mod.adopt_prewarmed_annotations(sid, msg_id, docs, cits)
            elapsed = time.perf_counter() - t0
        assert got == rects
        assert elapsed < 0.05
        assert "[PERF] join-block" in caplog.text
        with rp_mod._registry_lock:
            assert plan.docs_signature not in rp_mod._PREWARM_FUTURES
    finally:
        _cleanup(sid, msg_id, plan.docs_signature, fut)
