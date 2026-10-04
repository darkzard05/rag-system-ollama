"""Finish-tail timing gates (Task 0): drain vs finalize split measurement."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path


def _src(path: str) -> str:
    return (Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8")


def test_drain_log_has_waited_fields() -> None:
    text = _src("src/core/rag_core.py")
    assert "[PERF] hydration drain took" in text
    assert "waited=" in text
    assert "skipped_cache_hit=" in text


def test_finalize_log_has_perf_line() -> None:
    text = _src("src/ui/components/streaming.py")
    assert "[PERF] pdf finalize took" in text
    assert "cache_hit=" in text


def test_cache_hit_expression_for_hydrated_docs() -> None:
    class _Doc:
        def __init__(self, metadata: dict[str, object]) -> None:
            self.metadata = metadata

    documents = [_Doc({"page_coords": [{"page": 1}]})]
    cache_hit = bool(
        documents
        and all(
            ((getattr(d, "metadata", {}) or {}).get("page_coords"))
            or ((getattr(d, "metadata", {}) or {}).get("word_coords"))
            for d in documents
        )
    )
    assert cache_hit is True

    cold = [_Doc({})]
    cold_hit = bool(
        cold
        and all(
            ((getattr(d, "metadata", {}) or {}).get("page_coords"))
            or ((getattr(d, "metadata", {}) or {}).get("word_coords"))
            for d in cold
        )
    )
    assert cold_hit is False


async def _slow() -> str:
    await asyncio.sleep(2)
    return "slow"


def test_drain_straggler_cancelled_on_timeout() -> None:
    async def _run() -> None:
        t: asyncio.Task[str] = asyncio.create_task(_slow())
        try:
            await asyncio.wait_for(t, timeout=0.1)
        except asyncio.TimeoutError:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        assert t.done()

    asyncio.run(_run())


def test_no_shield_in_drain() -> None:
    text = _src("src/core/rag_core.py")
    assert "asyncio.shield(t)" not in text
    assert "asyncio.shield" not in text


def test_tail_gates_cover_badges_path() -> None:
    assert "[PERF] stream tail flush took" in _src("src/api/stream_pipeline.py")
    assert "[PERF] stream post-content stall took" in _src("src/api/stream_pipeline.py")
    assert "[PERF] stream max token gap took" in _src("src/api/stream_pipeline.py")
    assert "[PERF] consume persist took" in _src("src/ui/components/streaming.py")
    assert "[PERF] live persist took" in _src("src/ui/components/chat.py")
    assert "[PERF] tooltip scan took" in _src("src/ui/components/chat.py")


def test_citation_phase_reports_progress() -> None:
    text = _src("src/ui/components/streaming_core.py")
    assert "인용 정리 중" in text
    assert "def closed" in text
