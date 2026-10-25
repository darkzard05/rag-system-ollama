import asyncio
from contextlib import suppress
from typing import Any, cast

import pytest


async def mock_async_generator():
    try:
        await asyncio.sleep(0.5)
        yield "chunk1"
    finally:
        pass


@pytest.mark.asyncio
async def test_sse_generator_cancellation_drain():
    gen = mock_async_generator()
    iterator = gen.__aiter__()

    waiter = asyncio.create_task(cast(Any, iterator).__anext__())

    try:
        if waiter is not None and not waiter.done():
            waiter.cancel()
            with suppress(Exception):
                await asyncio.wait({waiter}, timeout=1.0)

        try:
            await cast(Any, gen).aclose()
        except Exception as e:
            pytest.fail(f"aclose failed with exception: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception during drainage: {e}")

    assert True
