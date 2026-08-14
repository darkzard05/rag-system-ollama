from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from core.session import SessionManager
from services.optimization.caching_optimizer import ObjectCache

logger = logging.getLogger(__name__)

# LRU/제거/통계 부속은 async ObjectCache 가 담당하며, EngineCacheManager는
# SessionManager 기반 저장소 + 루프/해시 검증을 수행하는 FACADE입니다.
# caller(rag_core.py, pipeline_builder.py)는 동기 호출(await 없음)이지만, 그
# 호출은 이미 실행 중인 이벤트 루프 안에서 일어날 수 있습니다. 따라서 async
# ObjectCache 는 별도 백그라운드 스레드의 전용 루프에서
# run_coroutine_threadsafe 로 구동합니다 (호출 루프를 방해하지 않음).
# 제약 R9: SyncCacheBridge 는 사용하지 않습니다(그것은 동기 VectorStoreCache 용).
_ENGINE_CACHE: ObjectCache[Any] = ObjectCache[Any](max_size=1000, ttl_seconds=0.0)

# async ObjectCache 를 구동하는 백그라운드 전용 스레드 + 루프.
_cache_thread: threading.Thread | None = None
_cache_loop: asyncio.AbstractEventLoop | None = None
_cache_lock = threading.Lock()


def _get_cache_loop() -> asyncio.AbstractEventLoop:
    """백그라운드 스레드의 전용 이벤트 루프를 생성/반환 (lazy, once)."""
    global _cache_loop, _cache_thread
    with _cache_lock:
        if _cache_loop is not None and not _cache_loop.is_closed():
            return _cache_loop
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name="EngineCache-ObjectCache", daemon=True
        )
        thread.start()
        _cache_loop = loop
        _cache_thread = thread
        return loop


def _run_async(coro: Any) -> Any:
    """백그라운드 루프에서 async ObjectCache 코루틴을 동기적으로 완료."""
    loop = _get_cache_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class EngineCacheManager:
    @staticmethod
    def _cache_key(session_id: str) -> str:
        return f"{session_id}"

    @staticmethod
    def get_engine(session_id: str) -> Any | None:
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        rag_engine = SessionManager.get("rag_engine", session_id=session_id)
        cached_loop_id = SessionManager.get(
            "rag_engine_loop_id", 0, session_id=session_id
        )
        cached_file_hash = SessionManager.get(
            "rag_engine_file_hash", session_id=session_id
        )
        current_file_hash = SessionManager.get("file_hash", session_id=session_id)

        if (
            not rag_engine
            or cached_loop_id != current_loop_id
            or cached_file_hash != current_file_hash
        ):
            if rag_engine:
                logger.info(
                    "[RAG] [ENGINE] 캐시 무효화 "
                    f"(loop: {cached_loop_id}->{current_loop_id}, "
                    f"file_hash: {cached_file_hash!r}->{current_file_hash!r})"
                )
            # 미러된 ObjectCache 에서도 제거(정합성). 실패해도 조회 결과엔 영향 없음.
            with contextlib.suppress(Exception):  # noqa: BLE001 - 방어적 무시
                _run_async(
                    _ENGINE_CACHE.delete(EngineCacheManager._cache_key(session_id))
                )
            return None

        logger.info("[RAG] [ENGINE] 캐시된 rag_engine 사용")
        return rag_engine

    @staticmethod
    def set_engine(session_id: str, engine: Any) -> None:
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        # 엔진이 참조하는 문서의 해시를 함께 기록해, 이후 file_hash가 바뀌면
        # get_engine이 이전 문서 기준 엔진을 반환하지 않게 합니다 (팬텀 상태 방지).
        file_hash = SessionManager.get("file_hash", session_id=session_id)
        SessionManager.set("rag_engine", engine, session_id=session_id)
        SessionManager.set("rag_engine_loop_id", current_loop_id, session_id=session_id)
        SessionManager.set("rag_engine_file_hash", file_hash, session_id=session_id)
        logger.info(f"[RAG] [ENGINE] 엔진 캐시됨 (loop_id={current_loop_id})")

        # LRU/제거 부속을 async ObjectCache 로 라우팅 (Facade 책임).
        try:
            _run_async(
                _ENGINE_CACHE.set(EngineCacheManager._cache_key(session_id), engine)
            )
        except Exception:  # noqa: BLE001 - 백엔드 장애가 메인 저장소를 침해하지 않게
            logger.warning(
                "[RAG] [ENGINE] ObjectCache 저장 실패 — SessionManager 저장은 유지",
                exc_info=True,
            )
