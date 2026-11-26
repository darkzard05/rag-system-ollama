"""An unhealthy cached connection must not be returned to callers."""

import asyncio
import time
from unittest.mock import AsyncMock

from cache.coord_cache import CoordCacheManager


class _IsolatedCoordCache(CoordCacheManager):
    """Subclass with its own singleton slot so the test never mutates the real one."""

    _instance = None


class _FakeConnection:
    def __init__(self) -> None:
        self.executed = False

    async def execute(self, sql: str, *args: object) -> None:  # noqa: ARG002
        self.executed = True
        if sql == "SELECT 1":
            raise RuntimeError("connection lost")

    async def close(self) -> None:
        return None


def _manager_with_unhealthy_conn() -> tuple[CoordCacheManager, _FakeConnection]:
    manager = _IsolatedCoordCache.__new__(_IsolatedCoordCache)
    manager._schema_ready = True
    manager._conn_lock = asyncio.Lock()
    broken = _FakeConnection()
    manager._conn = broken
    manager._conn_healthy = False
    manager._conn_created_at = time.monotonic()
    manager._last_health_check = time.monotonic()
    return manager, broken


async def test_unhealthy_connection_is_replaced_before_reuse() -> None:
    """A failed write must force a reconnect even inside the health interval."""
    manager, broken = _manager_with_unhealthy_conn()
    fresh = _FakeConnection()
    manager._connect = AsyncMock(return_value=fresh)  # type: ignore[method-assign]

    returned = await manager._get_connection()

    assert returned is fresh
    assert broken.executed is True
