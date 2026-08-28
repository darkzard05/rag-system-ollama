"""
TDD tests proving the coord-cache read-failure fix.

Before the fix, ``CoordCacheManager.get_coords`` / ``get_coords_batch`` silently
returned ``None`` / ``{}`` on any read error (DB corruption, I/O failure, blob
parse failure). After the fix they surface a ``CoordCacheReadError`` so the
caller (``document_hydrator``) can record a per-file failure instead of masking
it.

These tests exercise the REAL public wrapper behavior:
  * T1 forces the inner impl to raise -> the wrapper MUST raise
    ``CoordCacheReadError`` (previously it swallowed the error).
  * T2 simulates a legitimately-absent page (no cache entry) -> the wrapper
    MUST return ``None`` / ``{}`` WITHOUT raising.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cache.coord_cache import CoordCacheManager
from common.exceptions import CoordCacheReadError


class _RaisingImplError(RuntimeError):
    """Sentinel used to force the inner cache impl to blow up."""


async def _raising_batch_impl(
    self: CoordCacheManager, file_hash: str, page_nums: list[int]
) -> dict[int, list[dict[str, Any]]]:
    raise _RaisingImplError("injected batch read failure")


async def _raising_single_impl(
    self: CoordCacheManager, file_hash: str, page_num: int
) -> list[dict[str, Any]] | None:
    raise _RaisingImplError("injected single read failure")


@pytest.fixture
def manager() -> Any:
    """Fresh singleton instance with a synchronous, in-thread ``_submit``.

    We replace ``_submit`` so coroutines run directly in the current event loop
    (no owner-loop thread is started). This keeps tests deterministic and fast
    while still routing through the real public ``get_coords*`` wrappers.
    """
    CoordCacheManager._instance = None
    inst = CoordCacheManager()

    async def _sync_submit(coro: Any) -> Any:
        return await coro

    inst._submit = _sync_submit  # type: ignore[method-assign]
    yield inst
    CoordCacheManager._instance = None


def test_t1_get_coords_batch_raises_on_impl_failure(manager: Any) -> None:
    """T1: inner batch impl exception must surface as ``CoordCacheReadError``."""
    manager._get_coords_batch_impl = _raising_batch_impl  # type: ignore[method-assign]

    with pytest.raises(CoordCacheReadError):
        asyncio.run(manager.get_coords_batch("hashA", [1, 2]))


def test_t1_get_coords_raises_on_impl_failure(manager: Any) -> None:
    """T1: inner single-page impl exception must surface as ``CoordCacheReadError``."""
    manager._get_coords_impl = _raising_single_impl  # type: ignore[method-assign]

    with pytest.raises(CoordCacheReadError):
        asyncio.run(manager.get_coords("hashA", 1))


def test_t2_missing_page_returns_empty_without_raise(manager: Any) -> None:
    """T2: a legitimately-absent page returns ``{}`` / ``None`` and does NOT raise.

    The impl returns an empty result (no DB row) — the public wrapper must pass
    that through silently rather than错误地 raising.
    """

    async def _empty_batch_impl(
        *args: Any, **kwargs: Any
    ) -> dict[int, list[dict[str, Any]]]:
        return {}

    async def _empty_single_impl(
        *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]] | None:
        return None

    manager._get_coords_batch_impl = _empty_batch_impl  # type: ignore[method-assign]
    manager._get_coords_impl = _empty_single_impl  # type: ignore[method-assign]

    batch = asyncio.run(manager.get_coords_batch("missing", [1]))
    assert batch == {}

    single = asyncio.run(manager.get_coords("missing", 1))
    assert single is None
