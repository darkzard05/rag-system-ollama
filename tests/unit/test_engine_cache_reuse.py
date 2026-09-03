"""
EngineCacheManager 재사용/무효화 검증.

- 동일 file_hash: 엔진 재사용 (hit) — 이벤트 루프가 달라도 재사용
- file_hash 변경(팬텀 상태): 캐시 사용 안 함 (miss)
"""

import asyncio

from cache.engine_cache import EngineCacheManager
from core.session import SessionManager


def test_engine_reuse_without_hash():
    """file_hash 미설정 상태에서 set 직후 get → 엔진 재사용 (hit)."""
    sid = "engine_reuse_no_hash"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()

    async def _set_and_get():
        EngineCacheManager.set_engine(sid, engine)
        return EngineCacheManager.get_engine(sid)

    got = asyncio.run(_set_and_get())
    assert got is engine


def test_engine_reused_across_loops():
    """서로 다른 이벤트 루프에서 set/get 해도 엔진을 재사용한다.

    업로드 빌드(AsyncWorker 루프)와 쿼리 스트림(`asyncio.run` 새 루프)이 서로
    다른 루프이므로, 루프가 달라도 file_hash 가 같으면 캐시가 유효해야 한다.
    """
    sid = "engine_across_loops"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "hash_x", session_id=sid)
    engine = object()

    # asyncio.run()은 루프를 재활용해 id()가 같아질 수 있으므로,
    # 두 루프를 동시에 살려둔 채 교차 set/get 해야 확정적으로 검증됩니다.
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

    assert result is engine


def test_hit_on_same_hash():
    """동일 file_hash → 캐시 hit (저장된 엔진 반환)."""
    sid = "engine_hit_same_hash"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "hash_x", session_id=sid)
    engine = object()

    async def _run():
        EngineCacheManager.set_engine(sid, engine)
        assert EngineCacheManager.get_engine(sid) is engine

    asyncio.run(_run())


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


def test_miss_on_loop_change_without_hash():
    """file_hash 미설정(일치) 상태에서 루프만 달라도 엔진을 재사용한다.

    루프 변경은 무효화 사유가 아니다 — 문서 해시만 바뀌었을 때 miss 해야 한다.
    """
    sid = "engine_loop_no_hash_miss"
    SessionManager.reset()
    SessionManager.init_session(sid)
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
            # 새 루프에서 조회해도 루프 불일치는 miss 사유가 아니다.
            return EngineCacheManager.get_engine(sid)

        result = loop_b.run_until_complete(_get())
    finally:
        loop_a.close()
        loop_b.close()

    assert result is engine


def test_no_invalidation_log_on_same_hash_reuse(caplog):
    """동일 file_hash 재조회 시 '캐시 무효화' 로그가 없고 재사용 로그가 있다.

    문제 1 회귀 가드: 이전에는 루프 불일치로 file_hash 가 같아도 매번
    '캐시 무효화'가 출력·재빌드됐다. 이제 동일 해시면 캐시를 재사용한다.
    """
    import logging

    from cache.engine_cache import logger as engine_logger

    sid = "engine_log_regression"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "hash_c", session_id=sid)
    engine = object()

    async def _run():
        EngineCacheManager.set_engine(sid, engine)
        # 두 번째 조회가 무효화 없이 캐시를 재사용해야 한다.
        assert EngineCacheManager.get_engine(sid) is engine

    with caplog.at_level(logging.INFO, logger=engine_logger.name):
        asyncio.run(_run())

    logs = [r.getMessage() for r in caplog.records]
    assert "캐시 무효화" not in logs
    assert any("캐시된 rag_engine 사용" in m for m in logs)


def test_invalidation_logged_on_hash_change(caplog):
    """file_hash 변경 시에만 '캐시 무효화' 로그가 출력된다."""
    import logging

    from cache.engine_cache import logger as engine_logger

    sid = "engine_log_hash_change"
    SessionManager.reset()
    SessionManager.init_session(sid)
    engine = object()

    async def _run():
        SessionManager.set("file_hash", "hash_d", session_id=sid)
        EngineCacheManager.set_engine(sid, engine)
        SessionManager.set("file_hash", "hash_e", session_id=sid)
        assert EngineCacheManager.get_engine(sid) is None

    with caplog.at_level(logging.INFO, logger=engine_logger.name):
        asyncio.run(_run())

    logs = [r.getMessage() for r in caplog.records]
    assert any("캐시 무효화" in m for m in logs)
