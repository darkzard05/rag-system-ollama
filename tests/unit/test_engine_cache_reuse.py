"""
EngineCacheManager 재사용/무효화 검증 (문제 1 + Task 2 리팩터).

- 동일 이벤트 루프 + 동일 file_hash: 엔진 재사용 (hit)
- file_hash 변경(팬텀 상태): 캐시 사용 안 함 (miss)
- 루프 id 변경: 캐시 무효화 (miss)
- Task 2: LRU/제거 부속은 async ObjectCache(ObjectCache)로 라우팅된다.
"""

import asyncio

from cache.engine_cache import EngineCacheManager, _ENGINE_CACHE
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


def test_hit_on_same_loop_and_hash():
    """동일 루프 + 동일 file_hash → 캐시 hit (저장된 엔진 반환)."""
    sid = "engine_hit_same_loop_hash"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "hash_x", session_id=sid)
    engine = object()

    async def _run():
        EngineCacheManager.set_engine(sid, engine)
        assert EngineCacheManager.get_engine(sid) is engine

    asyncio.run(_run())
    # Facade는 ObjectCache에 미러링한다 (Task 2 라우팅 검증).
    assert _ENGINE_CACHE.get_stats().cache_size >= 1


def test_miss_on_changed_hash():
    """file_hash 변경(팬텀 상태) → 캐시 miss (None 반환, 재빌드 유도)."""
    sid = "engine_miss_changed_hash"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()

    async def _run():
        SessionManager.set("file_hash", "hash_a", session_id=sid)
        EngineCacheManager.set_engine(sid, engine)
        # 동일 해시 → 재사용
        assert EngineCacheManager.get_engine(sid) is engine
        # 문서 해시 변경 → 이전 엔진 반환 금지
        SessionManager.set("file_hash", "hash_b", session_id=sid)
        assert EngineCacheManager.get_engine(sid) is None

    asyncio.run(_run())


def test_miss_on_dead_loop_id():
    """루프 id가 죽은(바뀐) 세션 → 캐시 miss (None 반환)."""
    sid = "engine_miss_dead_loop"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "hash_z", session_id=sid)
    engine = object()

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop_a)

        async def _set():
            EngineCacheManager.set_engine(sid, engine)

        loop_a.run_until_complete(_set())

        asyncio.set_event_loop(loop_b)

        async def _get():
            # 새로운 루프에서 조회 → cached_loop_id 불일치로 miss
            return EngineCacheManager.get_engine(sid)

        result = loop_b.run_until_complete(_get())
    finally:
        loop_a.close()
        loop_b.close()

    assert result is None
