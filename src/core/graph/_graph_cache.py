"""그래프 캐시 인프라스트럭처 — 컴파일된 그래프/체크포인터의 프로세스 전역 캐시."""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, cast

from services.optimization.caching_optimizer import ObjectCache

logger = logging.getLogger(__name__)


@dataclass
class _CompiledGraphEntry:
    """통합 캐시에 보관되는 단일 그래프 항목 (컴파일 결과 + 체크포인터)."""

    compiled: Any = None
    checkpointer: Any = None


# 통합 캐시(ObjectCache)에 보관되는 항목의 키 — 프로세스 전역 단일 항목.
_GRAPH_CACHE_KEY = "compiled_graph"

# 통합 캐시 백엔드 — 컴파일된 그래프/체크포인터를 인메모리 객체로 보관.
# R8/R13: 이 백엔드는 LRU/제거/TTL 부속만 담당하며, 동시 빌드 보호는
# _GraphCache 프록시가 보유한 단일 asyncio.Lock + 이중 확인이 전담한다.
_graph_object_cache: ObjectCache[_CompiledGraphEntry] = ObjectCache[
    _CompiledGraphEntry
](max_size=1, ttl_seconds=0.0)


# ObjectCache는 async API이므로, 동기 호출 경로(테스트/삭제 콜백)에서는
# 전용 백그라운드 루프에서 run_coroutine_threadsafe로 구동한다
# (engine_cache.py의 SyncCacheBridge 대체 패턴과 동일).
# 그래프 캐시 전용 백그라운드 루프 상태(루프/락)를 단일 홀더에 캡슐화하여
# 모듈 전역 mutable 상태의 접근을 한 객체로 모읍니다 (테스트 용이성).
# _graph_object_cache(max_size=1)는 변경 없이 그대로 둔다.
@dataclass
class _GraphCacheLoopState:
    """그래프 캐시 전용 백그라운드 이벤트 루프 상태를 보관하는 모듈 전역 홀더."""

    loop: asyncio.AbstractEventLoop | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_graph_cache_loop_state = _GraphCacheLoopState()


def _get_graph_cache_loop() -> asyncio.AbstractEventLoop:
    """그래프 캐시 전용 백그라운드 이벤트 루프를 생성/반환 (lazy, once)."""
    with _graph_cache_loop_state.lock:
        current = _graph_cache_loop_state.loop
        if current is not None and not current.is_closed():
            return current

        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name="GraphCache-ObjectCache", daemon=True
        )
        thread.start()
        _graph_cache_loop_state.loop = loop
        return loop


def _run_graph_cache_coro(coro: Any) -> Any:
    """전용 루프에서 async ObjectCache 코루틴을 동기적으로 완료."""
    loop = _get_graph_cache_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class _GraphCache:
    """컴파일된 그래프, 체크포인터, 빌드 락을 안전하게 캡슐화하는 프록시.

    실제 항목은 통합 캐시(_graph_object_cache: ObjectCache)에 보관되며,
    본 프록시는 (1) 단일 전역 asyncio.Lock + 이중 확인 불변식을 보유하고,
    (2) 테스트/삭제 콜백이 직접 찌르는 동기 surface
    (.compiled/.checkpointer/.get_lock())를 노출한다.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None

    def get_lock(self) -> asyncio.Lock:
        """지연 초기화된 단일 그래프 빌드 락을 반환합니다.

        asyncio.Lock 은 최초 acquire 시점에 event loop 에 바인딩되므로, 생성 직후
        ``_loop`` 는 ``None`` 입니다. 테스트/워커처럼 함수 단위로 루프가 교체되는
        환경에서 캐시된 락을 무조건 재사용하면 ``bound to a different event loop``
        오류가 나므로, 이미 바인딩된 루프가 현재 루프와 다를 때만 새 락으로
        재생성합니다 (``_loop is None`` 인 미바인딩 락은 같은 객체로 재사용).
        """
        if self._lock is None or (
            # asyncio.Lock._loop is private; cast to Any to read the bound loop
            # (None until first acquire). Recreate if bound to a different loop.
            cast("Any", self._lock)._loop is not None
            and cast("Any", self._lock)._loop is not asyncio.get_event_loop()
        ):
            self._lock = asyncio.Lock()
        return self._lock

    def _get_entry(self) -> _CompiledGraphEntry | None:
        return _run_graph_cache_coro(_graph_object_cache.get(_GRAPH_CACHE_KEY))

    def _set_entry(self, entry: _CompiledGraphEntry) -> None:
        _run_graph_cache_coro(_graph_object_cache.set(_GRAPH_CACHE_KEY, entry))

    @property
    def compiled(self) -> Any:
        entry = self._get_entry()
        return entry.compiled if entry is not None else None

    @compiled.setter
    def compiled(self, value: Any) -> None:
        entry = self._get_entry() or _CompiledGraphEntry()
        entry.compiled = value
        self._set_entry(entry)

    @property
    def checkpointer(self) -> Any:
        """그래프에 연결된 체크포인터(saver)를 반환합니다."""
        entry = self._get_entry()
        return entry.checkpointer if entry is not None else None

    @checkpointer.setter
    def checkpointer(self, value: Any) -> None:
        entry = self._get_entry() or _CompiledGraphEntry()
        entry.checkpointer = value
        self._set_entry(entry)

    def invalidate(self) -> None:
        """컴파일된 그래프를 무효화하여 다음 build_graph() 호출 시 재컴파일합니다."""
        _run_graph_cache_coro(_graph_object_cache.delete(_GRAPH_CACHE_KEY))


_graph_cache = _GraphCache()


def invalidate_graph_cache() -> None:
    """Force recompilation of the LangGraph on the next build_graph() call."""
    _graph_cache.invalidate()


def delete_graph_thread(thread_id: str) -> None:
    """세션 종료 시 그래프 체크포인터의 해당 thread를 제거합니다 (R1a-02/R1b-02).

    InMemorySaver는 퇴거 정책이 없는 프로세스 전역 저장소이므로, 세션이 삭제될 때
    명시적으로 정리하지 않으면 thread_id(=session_id) 수만큼 체크포인트가 무제한
    누적된다. 체크포인트가 아직 구성되지 않았으면(그래프 미실행) 조용히 무시한다.
    """
    cp = _graph_cache.checkpointer
    if cp is None:
        logger.debug("[RAG] [GRAPH] 체크포인터 미구성 — thread 정리 생략")
        return
    cp.delete_thread(thread_id)
