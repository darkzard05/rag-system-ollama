"""Regression: ClientPool.get_async_client concurrent recreation race.

Root cause: src/core/resource_manager.py ClientPool.get_async_client used a
threading.Lock-protected sync path but an unguarded check-then-act + await
close() async path, so N concurrent host-switch callers each created a client.
Fix: asyncio.Lock serializes the slow path with double-checked locking.

Docs: Python asyncio docs require asyncio.Lock (not threading.Lock) to guard
shared state across await; threading.Lock only serializes OS threads.

Contract (relaxed on purpose): the only hard invariant is a single shared
object per loop+host. Creation count is informational (<=2) so future
pre-creation/caching changes don't break this test.
"""

import asyncio
from unittest.mock import patch

from core.resource_manager import ClientPool


class _FakeAsyncClient:
    created = 0

    def __init__(self, host: str = "") -> None:
        type(self).created += 1
        self.host = host
        self.closed = False

    async def close(self) -> None:
        await asyncio.sleep(0.01)
        self.closed = True


async def test_concurrent_host_switch_yields_single_client() -> None:
    pool = ClientPool()
    _FakeAsyncClient.created = 0
    with patch("ollama.AsyncClient", side_effect=_FakeAsyncClient):
        first = await pool.get_async_client("http://host-a:11434")
        assert _FakeAsyncClient.created == 1

        results = await asyncio.gather(
            *[pool.get_async_client("http://host-b:11434") for _ in range(10)]
        )

        ids = {id(c) for c in results}
        assert len(ids) == 1, f"race: {len(ids)} distinct clients returned"
        # Informational only: 1 initial + at most 1 recreation. Relaxed from
        # == 2 so pre-creation/caching refactors don't cause false failures.
        assert _FakeAsyncClient.created <= 2, (
            f"race: {_FakeAsyncClient.created} creations (expected <= 2)"
        )
        assert all(getattr(c, "host", "") == "http://host-b:11434" for c in results)
        assert first.closed is True
