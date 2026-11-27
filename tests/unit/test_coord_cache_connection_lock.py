"""A DB operation must keep its connection until the lock is released."""

import asyncio
import contextlib
import importlib
import time

import pytest

from cache.coord_cache import CoordCacheManager

coord_module = importlib.import_module("cache.coord_cache")


class _IsolatedCoordCache(CoordCacheManager):
    """Subclass with its own singleton slot so the real manager is untouched."""

    _instance = None


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def execute(self, sql: str, *args: object) -> None:  # noqa: ARG002
        if self.closed:
            raise ValueError("no active connection")

    async def commit(self) -> None:
        if self.closed:
            raise ValueError("no active connection")

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _manager() -> CoordCacheManager:
    manager = _IsolatedCoordCache.__new__(_IsolatedCoordCache)
    manager._schema_ready = True
    manager._conn_lock = asyncio.Lock()
    manager._db_lock = asyncio.Lock()
    manager._conn = _FakeConnection()
    manager._conn_healthy = True
    manager._conn_created_at = time.monotonic()
    manager._last_health_check = time.monotonic()

    async def connect() -> _FakeConnection:
        conn = _FakeConnection()
        manager._conn = conn
        manager._conn_healthy = True
        manager._conn_created_at = time.monotonic()
        manager._last_health_check = time.monotonic()
        return conn

    manager._connect = connect  # type: ignore[method-assign]
    return manager


async def test_connection_is_not_replaced_while_operation_holds_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contender must not close the connection of an in-flight operation."""
    monkeypatch.setattr(coord_module, "_CONNECTION_MAX_AGE", 0.0)
    manager = _manager()

    async def contender() -> None:
        async with manager._db_operation() as other_db:
            await other_db.execute("SELECT 1")

    task: asyncio.Task[None] | None = None
    try:
        async with manager._db_operation() as holder_db:
            task = asyncio.create_task(contender())
            await asyncio.sleep(0.01)
            assert holder_db.closed is False
            await holder_db.commit()
    finally:
        if task is not None and not task.done():
            task.cancel()
    if task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await task
