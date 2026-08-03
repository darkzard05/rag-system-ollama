"""
EngineCacheManager 루프 바인딩 검증 (문제 1).

동일 이벤트 루프에서는 엔진을 재사용하고, 루프가 변경되면 무효화해야 합니다.
"""

import asyncio

from cache.engine_cache import EngineCacheManager
from core.session import SessionManager


def test_engine_reuse_on_same_loop():
    sid = "engine_reuse_same_loop"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()

    async def _set_and_get():
        EngineCacheManager.set_engine(sid, engine)
        return EngineCacheManager.get_engine(sid)

    got = asyncio.run(_set_and_get())
    assert got is engine


def test_engine_invalidated_on_loop_change():
    sid = "engine_loop_change"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()

    # asyncio.run()은 루프를 재활용해 id()가 같아질 수 있으므로,
    # 두 루프를 동시에 살려둔 채 비교해야 확정적으로 검증됩니다.
    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop_a)

        async def _set():
            EngineCacheManager.set_engine(sid, engine)

        loop_a.run_until_complete(_set())

        asyncio.set_event_loop(loop_b)

        async def _get():
            return EngineCacheManager.get_engine(sid)

        result = loop_b.run_until_complete(_get())
    finally:
        loop_a.close()
        loop_b.close()

    assert result is None
