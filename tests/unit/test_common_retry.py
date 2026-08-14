"""common/retry.py 헬퍼 테스트.

retry_with_backoff (값 반환) 와 retry_stream (비동기 제너레이터) 를 검증합니다.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from common.retry import retry_stream, retry_with_backoff


# ---------------------------------------------------------------------------
# retry_with_backoff — sync value path
# ---------------------------------------------------------------------------


def test_retry_sync_success_first_try() -> None:
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert retry_with_backoff(flaky, max_retries=3) == "ok"
    assert calls == 1


def test_retry_sync_retry_then_success() -> None:
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("boom")
        return "ok"

    assert retry_with_backoff(flaky, max_retries=3, base_delay=0.0) == "ok"
    assert calls == 3


def test_retry_sync_exhaust_then_raise() -> None:
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        retry_with_backoff(
            flaky,
            max_retries=3,
            base_delay=0.0,
            retry_on=(ValueError,),
        )
    assert calls == 3


def test_retry_sync_uses_time_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync 경로는 asyncio.sleep 이 아니라 time.sleep 을 써야 합니다."""
    slept: list[float] = []
    monkeypatch.setattr("common.retry.time.sleep", lambda s: slept.append(s))
    async_slept: list[float] = []
    monkeypatch.setattr(
        "common.retry.asyncio.sleep",
        lambda s: async_slept.append(s),  # type: ignore[assignment]
    )
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ValueError("boom")
        return "ok"

    assert retry_with_backoff(flaky, max_retries=3, base_delay=1.0) == "ok"
    assert slept == [1.0]
    assert async_slept == []


# ---------------------------------------------------------------------------
# retry_with_backoff — async value path
# ---------------------------------------------------------------------------


async def test_retry_async_success_first_try() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry_with_backoff(flaky, max_retries=3) == "ok"
    assert calls == 1


async def test_retry_async_retry_then_success() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("boom")
        return "ok"

    assert await retry_with_backoff(flaky, max_retries=3, base_delay=0.0) == "ok"
    assert calls == 3


async def test_retry_async_exhaust_then_raise() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await retry_with_backoff(
            flaky, max_retries=2, base_delay=0.0, retry_on=(ValueError,)
        )
    assert calls == 2


# ---------------------------------------------------------------------------
# retry_stream — async generator (yields, not returns)
# ---------------------------------------------------------------------------


def _make_factory(
    items: list[str], errors: list[Exception | None]
) -> tuple[callable, list[int]]:
    """호출될 때마다 items 를 yield, errors[i] 가 있으면 i번째 반복에서 발생."""
    state = {"build": 0}

    def factory() -> asyncio.AsyncIterator[str]:
        build = state["build"]
        state["build"] += 1

        async def gen() -> asyncio.AsyncIterator[str]:
            for it in items:
                yield it
            err = errors[build] if build < len(errors) else None
            if err is not None:
                raise err

        return gen()

    return factory, state  # type: ignore[return-value]


async def test_retry_stream_yields_items() -> None:
    factory, _ = _make_factory(["a", "b", "c"], [None])

    out = [item async for item in retry_stream(factory, max_retries=3)]
    assert out == ["a", "b", "c"]


async def test_retry_stream_retry_then_success() -> None:
    """첫 항목 전에 오류가 나면 재시도하고, 성공 시 항목을 한 번만 yield."""

    state: dict[str, int] = {"build": 0}

    def factory() -> asyncio.AsyncIterator[str]:
        state["build"] += 1
        build = state["build"]

        async def gen() -> asyncio.AsyncIterator[str]:
            if build == 1:
                raise ConnectionError("drop before first")
            yield "a"
            yield "b"

        return gen()

    out = [item async for item in retry_stream(factory, max_retries=3, base_delay=0.0)]
    # 첫 빌드는 드롭 → 재시도 → 두 번째 빌드는 성공, 항목 중복 없음.
    assert out == ["a", "b"]
    assert state["build"] == 2


async def test_retry_stream_error_after_first_item_reraises() -> None:
    """첫 항목 이후 오류는 재시도하지 않고 재발생 (중복 전송 방지)."""
    state = {"build": 0}

    def factory() -> asyncio.AsyncIterator[str]:
        build = state["build"]
        state["build"] += 1

        async def gen() -> asyncio.AsyncIterator[str]:
            yield "a"
            if build == 0:
                raise ConnectionError("drop after first")

        return gen()

    with pytest.raises(ConnectionError, match="drop after first"):
        async for _ in retry_stream(factory, max_retries=3, base_delay=0.0):
            pass
    # 재시도 없이 한 번만 빌드되었어야 함.
    assert state["build"] == 1


async def test_retry_stream_exhaust_then_raise() -> None:
    factory, _ = _make_factory([], [ConnectionError("d1"), TimeoutError("d2")])

    with pytest.raises((ConnectionError, TimeoutError)):
        async for _ in retry_stream(factory, max_retries=2, base_delay=0.0):
            pass


async def test_retry_stream_cancelled_reraises() -> None:
    def factory() -> asyncio.AsyncIterator[str]:
        async def gen() -> asyncio.AsyncIterator[str]:
            yield "a"
            raise asyncio.CancelledError()

        return gen()

    with pytest.raises(asyncio.CancelledError):
        async for _ in retry_stream(factory, max_retries=3, base_delay=0.0):
            pass


async def test_retry_stream_handles_httpx_errors() -> None:
    state = {"build": 0}

    def factory() -> asyncio.AsyncIterator[str]:
        build = state["build"]
        state["build"] += 1

        async def gen() -> asyncio.AsyncIterator[str]:
            yield "x"
            if build == 0:
                raise httpx.TimeoutException("httpx drop")

        return gen()

    with pytest.raises(httpx.TimeoutException):
        async for _ in retry_stream(factory, max_retries=3, base_delay=0.0):
            pass
    assert state["build"] == 1
