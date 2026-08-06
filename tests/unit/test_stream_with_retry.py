import asyncio

import pytest

from core.rag_core import _stream_with_retry


async def test_mid_stream_error_does_not_replay():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        yield {"type": "content", "text": "first"}
        raise ConnectionError("mid-stream failure")

    with pytest.raises(ConnectionError):
        async for _ in _stream_with_retry(factory, max_retries=3, base_delay=0):
            pass
    assert calls["n"] == 1  # 재시도 없음 (중복 재생 방지)


async def test_pre_first_item_error_is_retried():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("first attempt fails")
        yield {"type": "content", "text": "ok"}

    items = [
        item async for item in _stream_with_retry(factory, max_retries=3, base_delay=0)
    ]
    assert calls["n"] == 2
    assert items == [{"type": "content", "text": "ok"}]


async def test_retry_exhaustion_still_raises():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise TimeoutError("always fails")
        yield

    with pytest.raises(TimeoutError):
        async for _ in _stream_with_retry(factory, max_retries=2, base_delay=0):
            pass
    assert calls["n"] == 2


async def test_cancelled_error_not_swallowed():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise asyncio.CancelledError()
        yield

    with pytest.raises(asyncio.CancelledError):
        async for _ in _stream_with_retry(factory, max_retries=3, base_delay=0):
            pass
    assert calls["n"] == 1
