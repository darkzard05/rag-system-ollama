"""
Verification tests for identified defects.
"""

import asyncio

import pytest
from src.core.session import SessionManager


@pytest.mark.asyncio
async def test_p0_2_session_sync_concurrency(session_context):
    """P0-2: 세션 동기화 및 스레드 안전성 검증"""

    async def worker(i):
        SessionManager.set(f"key_{i}", f"val_{i}", session_id=session_context)
        return SessionManager.get(f"key_{i}", session_id=session_context)

    results = await asyncio.gather(*(worker(i) for i in range(100)))

    for i, res in enumerate(results):
        assert res == f"val_{i}"
