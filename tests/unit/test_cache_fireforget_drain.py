"""Task2: deferred backends — fire-and-forget cache + concurrent hydration drain (TDD).

Failing-first coverage for:
  (a) cache-set off the critical path when QUERY_CACHE_ENABLED, with zero
      thread/zero embed overhead when disabled (default False);
  (b) hydration drain concurrent + order-preserving with fail-loud metrics;
  (c) lazy perf report (monitoring never breaks the answer) and no retriever
      object persistence from rag_core (pipeline_builder 504-511 anti-pattern).
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import core.graph._generate as gen_mod
from core.graph._generate import _schedule_query_cache_set
from core.rag_core import _drain_hydration_tasks, _lazy_perf_report

_MSG = "task2-cache-msg"


def _payload(**over: object) -> dict:
    base: dict = {
        "query": "q-cache",
        "response": '{"final_answer": "ok", "citations": []}',
        "confidence": 0.95,
        "file_hash": "fh-task2",
        "msg_id": _MSG,
    }
    base.update(over)
    return base


# --- (a) fire-and-forget cache-set -------------------------------------------


def test_cache_set_disabled_has_zero_thread_zero_embed_overhead() -> None:
    """QUERY_CACHE_ENABLED=False면 스레드/임베더/캐시 접근이 전혀 없다."""
    assert gen_mod.QUERY_CACHE_ENABLED is False
    with (
        patch("threading.Thread", side_effect=AssertionError("no thread")),
        patch(
            "core.graph._generate._ensure_query_cache_embedder",
            side_effect=AssertionError("no embed"),
        ),
        patch(
            "core.graph._generate.get_cache_manager",
            side_effect=AssertionError("no cache"),
        ),
    ):
        assert _schedule_query_cache_set(**_payload()) is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cache_set_enabled_is_fire_and_forget_off_critical_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """enabled면 set을 백그라운드로 예약하고 호출자는 기다리지 않고 복귀한다."""
    release = asyncio.Event()
    done = asyncio.Event()
    calls: list[str] = []

    async def _slow_set(*args: object, **kwargs: object) -> None:
        calls.append("set")
        await release.wait()
        done.set()

    with (
        patch.object(gen_mod, "QUERY_CACHE_ENABLED", True),
        patch(
            "core.graph._generate._ensure_query_cache_embedder",
            new=AsyncMock(),
        ),
        patch(
            "core.graph._generate.get_cache_manager",
        ) as manager_factory,
    ):
        manager_factory.return_value.set = _slow_set
        with caplog.at_level(logging.INFO):
            scheduled_at = time.perf_counter()
            assert _schedule_query_cache_set(**_payload()) is True  # type: ignore[arg-type]
            scheduled_took = time.perf_counter() - scheduled_at
        assert scheduled_took < 1.0, "예약 자체가 느린 set을 기다리면 안 된다"
        for _ in range(200):
            if calls:
                break
            await asyncio.sleep(0.01)
        assert calls == ["set"], "백그라운드 set이 예약되어야 한다"
        assert any(
            _MSG in rec.message and "cache-set" in rec.message for rec in caplog.records
        ), "msg_id가 [PERF] 예약 로그에 남아야 한다"
        release.set()
        await asyncio.wait_for(done.wait(), timeout=5.0)


@pytest.mark.asyncio
async def test_cache_set_below_min_confidence_never_scheduled() -> None:
    """신뢰도 미달이면 enabled여도 예약하지 않는다."""
    with (
        patch.object(gen_mod, "QUERY_CACHE_ENABLED", True),
        patch(
            "core.graph._generate.get_cache_manager",
            side_effect=AssertionError("no cache"),
        ),
    ):
        assert (
            _schedule_query_cache_set(**_payload(confidence=0.1))  # type: ignore[arg-type]
            is False
        )


# --- (b) concurrent hydration drain ------------------------------------------


@pytest.mark.asyncio
async def test_hydration_drain_is_concurrent_and_order_preserving(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """드레인이 순차가 아닌 동시 대기이며 결과 순서를 보존한다."""

    async def _slow(i: int) -> int:
        await asyncio.sleep(0.2)
        return i

    tasks = [asyncio.create_task(_slow(i)) for i in range(4)]
    with caplog.at_level(logging.INFO):
        started = time.perf_counter()
        drain = await _drain_hydration_tasks(tasks, timeout=5.0, msg_id=_MSG)
        took = time.perf_counter() - started
    assert drain.results == [0, 1, 2, 3], "gather 순서를 보존해야 한다"
    assert drain.failed == 0
    assert drain.timed_out is False
    assert took < 0.8, f"순차 드레인(0.8s)이면 안 된다: {took:.3f}s"
    assert any(
        _MSG in rec.message and "hydration drain" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_hydration_drain_timeout_cancels_pending_fail_loud(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """타임아웃 시 pending을 취소하고 실패를 로그+상태로 알린다."""

    async def _hang() -> None:
        await asyncio.sleep(30.0)

    tasks = [asyncio.create_task(_hang()) for _ in range(2)]
    with caplog.at_level(logging.ERROR):
        drain = await _drain_hydration_tasks(tasks, timeout=0.1, msg_id=_MSG)
    assert drain.failed == 2
    assert drain.timed_out is True
    assert all(t.done() for t in tasks)
    assert any("타임아웃" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_hydration_drain_counts_failures_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """실패 태스크는 세고 스트림 실패로 승격시키지 않는다."""

    async def _boom() -> None:
        raise RuntimeError("hydrate-broken")

    async def _ok() -> str:
        return "fine"

    tasks = [asyncio.create_task(_boom()), asyncio.create_task(_ok())]
    with caplog.at_level(logging.ERROR):
        drain = await _drain_hydration_tasks(tasks, timeout=5.0, msg_id=_MSG)
    assert drain.failed == 1
    assert drain.timed_out is False
    assert drain.results[1] == "fine"


# --- (c) lazy perf report + no retriever persistence --------------------------


def test_lazy_perf_report_never_breaks_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """generate_report 실패는 빈 dict로 흡수된다 (모니터링 lazy)."""
    with (
        patch(
            "services.monitoring.performance_monitor.get_performance_monitor",
            side_effect=RuntimeError("monitor down"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        assert _lazy_perf_report() == {}


def test_rag_core_never_persists_retriever_objects() -> None:
    """rag_core에 pipeline_builder 504-511식 retriever 영속 패턴이 없다."""
    import pathlib

    src = pathlib.Path("src/core/rag_core.py").read_text(encoding="utf-8")
    for key in (
        "active_faiss_retriever",
        "active_bm25_retriever",
        "active_retriever_hash",
    ):
        assert key not in src, f"rag_core가 retriever 영속 키를 쓰면 안 된다: {key}"
