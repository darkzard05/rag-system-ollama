"""
LangGraph를 사용하여 자가 교정(Self-Correction) RAG 워크플로우를 구성합니다.
의도 분류, 캐시 확인, 하이브리드 검색, 문서 평가, 쿼리 재구성, 생성의 단계를 포함합니다.
"""

import asyncio
import contextvars
import copy
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from api.schemas import AggregatedSearchResult, AnswerStructure, GraphState
from common.config import (
    ANALYSIS_PROTOCOL,
    DEFAULT_EMBEDDING_MODEL,
    GRADING_CONFIG,
    GRADING_ENABLED,
    MAX_CONCURRENT_INFERENCE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    PROMPT_TEMPLATES_CONFIG,
    QUERY_CACHE_ENABLED,
    QUERY_CACHE_MIN_CONF,
    QUERY_CACHE_TTL,
)
from common.utils import count_tokens_rough, fast_hash
from core.model_loader import ModelManager
from core.session import SessionManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization.caching_optimizer import ObjectCache, get_cache_manager

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Wave 3 grade reduction — memoization contract (T8).
# MUST stay byte-equal to ``GRADE_MEMO_KEY`` in ``core/pipeline_builder.py``
# (line ~41). We deliberately use the literal instead of importing it because
# pipeline_builder imports ``build_graph`` from this module at module scope,
# so ``from core.pipeline_builder import GRADE_MEMO_KEY`` would be a circular
# import. pipeline_builder owns the canonical definition + re-index invalidation;
# this module only stores into the same key.
_GRADE_MEMO_KEY = "grade_decision_memo"


def _grade_memo_key(state: Any, docs: list) -> str:
    """안정 해시로 메모 키를 생성합니다 (query + 정렬된 doc_id 집합).

    동일 (query, doc-set) 조합이면 항상 동일 키 → 같은 세션 내 반복 질의 시
    LLM 호출을 건너뛰고 이전 판단을 재사용한다. doc_id가 하나라도 바뀌면
    해시가 달라져(staleness miss) 오래된 메모가 재사용되지 않는다.
    """
    query = get_state_attr(state, "input") or ""
    # 재구성된 쿼리(search_queries[-1])가 실제 평가에 쓰인 경우 그걸 사용.
    search_queries = get_state_attr(state, "search_queries")
    if search_queries:
        query = search_queries[-1]
    doc_ids = sorted(_doc_stable_id(d) for d in docs)
    return fast_hash(f"GRADE|{query}|" + "|".join(doc_ids))


def _store_grade_memo(state: Any, docs: list, decision: dict, sid: str) -> None:
    """실제 LLM 판단을 세션 메모(dict)에 저장합니다.

    실패해도 채점 흐름을 깨뜨리지 않도록 내부에서 흡수한다 (최대한 best-effort).
    저장 실패 시 다음 반복에서 단순히 miss → LLM 경로로 폴백한다.
    """
    try:
        memo_key = _grade_memo_key(state, docs)
        existing = SessionManager.get(_GRADE_MEMO_KEY, session_id=sid, default={})
        if not isinstance(existing, dict):
            existing = {}
        existing[memo_key] = decision
        SessionManager.set(_GRADE_MEMO_KEY, existing, session_id=sid)
    except Exception as e:  # noqa: BLE001 - 메모 저장 실패는 치명적이지 않음
        logger.debug(f"[RAG] [GRADE] 메모 저장 실패 (무시): {e}")


# ============================================================================
# Per-stage latency tracing (Wave 1)
# ----------------------------------------------------------------------------
# Every query emits ONE consolidated `[QUERY][TIMING]` line aggregating the
# duration (ms) of each pipeline stage. Single-threaded LangGraph node
# execution means at most one node runs at a time, so a module-level
# threading.local buffer keyed by query lifecycle is safe: `preprocess`
# (always-runs-first node) resets it at the start of every query, and
# `generate` (terminal node) reads it once and emits the line before each
# return. The buffer is captured into a local at generate entry so retries
# (which re-run retrieve/grade) accumulate correctly.
# ============================================================================

_stage_timing_local = threading.local()


def _reset_stage_timings() -> None:
    """Clear per-query stage timing buffer (called at preprocess start)."""
    _stage_timing_local.stages = {
        "preprocess_ms": 0.0,
        "retrieve_ms": 0.0,
        "grade_ms": 0.0,
        "generate_total_ms": 0.0,
        "ttft_ms": 0.0,
    }


def _add_stage_ms(stage: str, ms: float) -> None:
    """Accumulate a stage duration into the per-query buffer."""
    if not hasattr(_stage_timing_local, "stages"):
        _reset_stage_timings()
    _stage_timing_local.stages[stage] = _stage_timing_local.stages.get(
        stage, 0.0
    ) + float(ms)


def _enter_stage(operation_type: OperationType, **metadata: Any) -> Any:
    """Begin a tracked operation, returning the OperationTracker context manager.

    Mirrors the existing ``with get_performance_monitor().track_operation(...)``
    pattern used in ``chunking.py`` but exposes the tracker so callers can exit
    it without re-indenting large node bodies.
    """
    return get_performance_monitor().track_operation(operation_type, dict(metadata))


def _emit_query_timing(timings: dict[str, float]) -> None:
    """Emit the single consolidated per-query timing line."""
    logger.info(
        f"[QUERY][TIMING] preprocess_ms={timings.get('preprocess_ms', 0.0):.1f} "
        f"retrieve_ms={timings.get('retrieve_ms', 0.0):.1f} "
        f"grade_ms={timings.get('grade_ms', 0.0):.1f} "
        f"generate_total_ms={timings.get('generate_total_ms', 0.0):.1f} "
        f"ttft_ms={timings.get('ttft_ms', 0.0):.1f}"
    )


def _get_session_id(config: RunnableConfig | None = None) -> str:
    """Extract session_id from RunnableConfig or fall back to current context."""
    if config and "configurable" in config:
        sid = config["configurable"].get("session_id")
        if sid:
            return sid
        # config는 전달됐지만 session_id가 누락된 경우 — 전파 버그 신호.
        # 정상적인 "default" 세션 사용(비 Streamlit 모드)은 조용히 폴백합니다.
        logger.warning("[GRAPH] config에 session_id 누락 — 암묵적 세션 폴백")
    return SessionManager.get_session_id()


# ============================================================================
# PHASE 2 — Speculative / eager generation overlapping grade_documents.
#
# `grade_documents` is a *routing* node (it decides the LangGraph route:
# `generate` vs `transform`/`rewrite`). We must NOT fuse it into `generate`.
# Instead, when grade is about to make its LLM call (the slow part), we *also*
# kick off the `generate` LLM stream eagerly. If grade's route is `generate`
# (the common case), `generate` is already warm → near-zero added TTFT (the
# grade duration is hidden behind the generation, effectively removing one
# round-trip from p99).
#
# If grade's route is `transform`/`rewrite`, the speculative stream is
# cancelled and its output is NEVER surfaced to the user.
#
# Safety constraints:
#  * We only overlap when MAX_CONCURRENT_INFERENCE > 1, so the two LLM calls
#    can genuinely run concurrently. With bound==1 (the default) the semaphore
#    would serialize them, so we skip overlap entirely — the pipeline behaves
#    exactly as before (no regression, no inverted ordering).
#  * During the speculative run, all `adispatch_custom_event` calls are
#    buffered (not emitted) so no partial answer reaches the UI. The buffer is
#    kept in a ContextVar so it never crosses threads/coroutines.
#  * The real `generate` node ADOPTS the already-running task: it replays the
#    buffered events and returns the task's result — a single LLM call, just
#    started earlier. On `transform`, that task is cancelled and the buffer
#    dropped.
# ============================================================================

# Buffered events for the in-flight speculative generate of the current query.
_spec_generate_events: "contextvars.ContextVar[list[_SpecEvent]]" = (
    contextvars.ContextVar("_spec_generate_events")
)


@dataclass
class _SpecEvent:
    """A single buffered adispatch_custom_event payload."""

    name: str
    data: dict[str, Any]
    config: RunnableConfig


@dataclass
class _SpecGenerate:
    """In-flight speculative generate for one thread_id."""

    task: "asyncio.Task[dict[str, Any]]"
    buffer: list[_SpecEvent]
    adopter: str | None = None  # thread_id that adopted the task, prevents reuse


# Per-thread_id registry of the currently speculative generate task. Populated
# by grade_documents, consumed (adopted or cancelled) by generate.
_spec_registry: dict[str, _SpecGenerate] = {}


def _spec_overlap_enabled() -> bool:
    """True iff two LLM calls can genuinely run concurrently this session."""
    return MAX_CONCURRENT_INFERENCE > 1


async def _dispatch_event(
    name: str,
    data: dict[str, Any],
    *,
    writer: StreamWriter | None,
    config: RunnableConfig,
) -> None:
    """Emit a custom event, or buffer it if a speculative generate is active.

    Buffering ensures a speculative (possibly-to-be-discarded) generate never
    leaks partial output to the user. The buffer is scoped to the current
    coroutine context via a ContextVar.
    """
    buf = _spec_generate_events.get(None)
    if buf is not None:
        buf.append(_SpecEvent(name=name, data=data, config=config))
        return
    await adispatch_custom_event(name, data, config=config)


def _start_speculative_generate(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> str | None:
    """Begin an eager generate task inside the current coroutine context.

    Returns the thread_id key under which the task is registered, or ``None``
    if overlap is disabled (bound==1) or no thread_id is available. The
    speculative generate's events are buffered until the route is decided.
    """
    if not _spec_overlap_enabled():
        return None
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    if thread_id in _spec_registry:
        # One speculative generate per query; ignore duplicate triggers.
        return None

    _spec_generate_events.set([])
    task = asyncio.ensure_future(generate(state, config, writer=writer))
    # Tag the task so the speculative generate instance does NOT adopt itself
    # (it must run normally and buffer its events instead).
    task._is_speculative = True  # type: ignore[attr-defined]
    _spec_registry[thread_id] = _SpecGenerate(task=task, buffer=[])
    # The speculative task buffers into the ContextVar list we just captured.
    _spec_registry[thread_id].buffer = _spec_generate_events.get([])
    logger.info("[RAG] [SPEC] eager generate 시작 (route 결정 대기)")
    return thread_id


def _adopt_speculative_generate(
    config: RunnableConfig,
) -> tuple["asyncio.Task[dict[str, Any]]", list[_SpecEvent]] | None:
    """Adopt a warm speculative generate task, if one exists for this thread_id.

    Marks it adopted so it cannot be reused. Returns ``(task, buffered_events)``
    for the adopting node to replay, or ``None`` if there is nothing to adopt
    (normal sequential path).
    """
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    spec = _spec_registry.pop(thread_id, None)
    if spec is None:
        return None
    if spec.adopter is not None:
        return None
    spec.adopter = thread_id
    return spec.task, spec.buffer


def _cancel_speculative_generate(config: RunnableConfig) -> None:
    """Cancel any speculative generate for this thread_id and drop its buffer.

    Used when grade routes to transform/rewrite — the speculative output must
    never reach the user.
    """
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    spec = _spec_registry.pop(thread_id, None)
    if spec is None:
        return None
    if spec.adopter is not None:
        return None
    spec.task.cancel()
    spec.buffer.clear()
    logger.info("[RAG] [SPEC] route=transform → speculative generate 취소 (미노출)")


def _replay_spec_events(
    buffer_token: "contextvars.Token[list[_SpecEvent]]",
) -> list[_SpecEvent]:
    """Return the buffered speculative events and clear them from the context."""
    events = _spec_generate_events.get([])
    _spec_generate_events.reset(buffer_token)
    return events


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
        """지연 초기화된 단일 그래프 빌드 락을 반환합니다 (매 호출 동일 객체)."""
        if self._lock is None:
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
    누적된다. 체크포인터가 아직 구성되지 않았으면(그래프 미실행) 조용히 무시한다.
    """
    cp = _graph_cache.checkpointer
    if cp is None:
        logger.debug("[RAG] [GRAPH] 체크포인터 미구성 — thread 정리 생략")
        return
    cp.delete_thread(thread_id)


def get_state_attr(state: Any, key: str, default: Any = None) -> Any:
    """dict와 object(GraphState) 모두에서 속성을 안전하게 가져옵니다."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


# 쿼리 응답 캐시 저장값 구조: {"response": str, "confidence": float}
# SemanticCache.get()은 내부 유사도 임계값을 통과한 entry.value(저장값) 또는 None만 반환하며,
# 유사도 점수나 히트 객체를 노출하지 않으므로 신뢰도는 저장 시 함께 직렬화한다.
_QUERY_CACHE_VALUE_VERSION = "1.0"


class _AsyncEmbeddingWrapper:
    """동기형 embed_query를 비동기 인터페이스로 감싸는 어댑터.

    SemanticCache._embed()는 self.embedding_model.embed_query(text)를
    무조건 await한다. Ollama 임베더(_NoTruncateOllamaEmbeddings)의 embed_query는
    동기형(list 반환)이라 await 시 crash한다. 캐시 소스는 건드리지 않고, 쿼리
    경로에서 캐시에 주입하는 임베더만 비동기 퍼사드로 감싸 호환시킨다.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    async def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._embedder, "embed_documents"):
            return self._embedder.embed_documents(texts)
        return [self._embedder.embed_query(t) for t in texts]


async def _ensure_query_cache_embedder() -> None:
    """전역 캐시 싱글톤의 세맨틱 캐시에 임베더를 주입한다.

    SemanticCache는 self.embedding_model이 None이면 get/set 시 항상 None을
    반환한다(임베딩 불가). 청커는 별도 CacheManager 인스턴스를 만들므로 전역
    싱글톤의 semantic_cache.embedding_model은 기본 비어 있다 — 쿼리 경로에서
    캐시를 쓰려면 런타임에 주입해야 한다. (캐시 클래스는 수정하지 않음)
    동기형 Ollama 임베더는 _AsyncEmbeddingWrapper로 감싸 await 호환을 맞춘다.
    """
    if not QUERY_CACHE_ENABLED:
        return
    cm = get_cache_manager()
    if cm.semantic_cache is None:
        return
    if cm.semantic_cache.embedding_model is not None:
        return
    try:
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
        cm.semantic_cache.embedding_model = _AsyncEmbeddingWrapper(embedder)
    except Exception as e:  # noqa: BLE001 - 캐시 누락은 치명적이지 않음
        logger.warning(f"[RAG] [CACHE] 임베더 주입 실패 — 쿼리 캐시 비활성화: {e}")


def _sanitize_channel_value(value: Any) -> Any:
    """상태 채널 값을 msgpack 직렬화 가능한 순수 타입으로 위생화합니다.

    R1a-05: JsonPlusSerializer(pickle_fallback=False) 전환에 따라 상태에 저장되는
    값은 int/str/float/bool/None/list/dict만 허용한다. 그 외 객체(커스텀 클래스
    인스턴스 등)를 만나면 조용히 pickle로 강등하지 않고 명시적 예외를 던진다.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_channel_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_channel_value(item) for item in value)
    if isinstance(value, dict):
        return {k: _sanitize_channel_value(v) for k, v in value.items()}
    raise ValueError(
        f"직렬화 불가 객체 감지 (pickle 강등 금지): {type(value).__name__} — "
        "채널에는 순수 타입만 저장할 수 있습니다."
    )


async def _safe_invoke(llm: Any, prompt: str, config: dict) -> Any:
    """Safely invoke an LLM with async fallback."""
    res = llm.ainvoke(prompt, config=config)
    if asyncio.iscoroutine(res):
        return await res
    return res


async def preprocess(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 수행합니다."""
    # [TIMING] 매 쿼리 시작 시 스테이지 누적 버퍼 초기화 (preprocess는 항상 최초 실행)
    _reset_stage_timings()
    _preprocess_op = _enter_stage(OperationType.QUERY_PROCESSING)
    _preprocess_op.__enter__()
    _preprocess_start = time.perf_counter()
    query = get_state_attr(state, "input", "").strip()
    logger.info(f"[RAG] [PREPROCESS] 입력 질의: '{query}'")

    import re

    from common.config import DYNAMIC_WEIGHTING_CONFIG, ENSEMBLE_WEIGHTS

    # 1. 의도 분류 및 동적 가중치 결정
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    intent = "rag"

    if len(query) < 10 and any(
        g in query.lower() for g in ["안녕", "hi", "hello", "반가워", "누구"]
    ):
        intent = "general"
        logger.info("[RAG] [PREPROCESS] 일상 대화(General) 의도 감지")

    if DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):
        keyword_patterns = DYNAMIC_WEIGHTING_CONFIG.get("keyword_patterns", [])
        is_keyword_heavy = any(re.search(p, query) for p in keyword_patterns)

        semantic_keywords = DYNAMIC_WEIGHTING_CONFIG.get("semantic_keywords", [])
        is_semantic_heavy = any(k in query for k in semantic_keywords)

        if is_keyword_heavy and not is_semantic_heavy:
            kw_w = DYNAMIC_WEIGHTING_CONFIG.get("keyword_weight", 0.8)
            weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}
            logger.info(f"[RAG] [PREPROCESS] 키워드 중심 질의 판단 (BM25: {kw_w})")
        elif is_semantic_heavy and not is_keyword_heavy:
            sm_w = DYNAMIC_WEIGHTING_CONFIG.get("semantic_weight", 0.8)
            weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}
            logger.info(f"[RAG] [PREPROCESS] 의미 중심 질의 판단 (FAISS: {sm_w})")

    _preprocess_ms = (time.perf_counter() - _preprocess_start) * 1000
    _add_stage_ms("preprocess_ms", _preprocess_ms)
    _preprocess_op.__exit__(None, None, None)

    # 쿼리 응답 캐시 조회 (선택적 opt-in). 세션에 활성 인덱싱 문서가 있을 때만
    # 후보군으로 삼는다 — 문서 없는 세션의 캐시 히트는 부정확하므로 무시한다.
    is_cached = False
    cached_response: str | None = None
    if QUERY_CACHE_ENABLED:
        sid = _get_session_id(config)
        has_doc = bool(SessionManager.get("file_hash", session_id=sid, default=None))
        if has_doc:
            await _ensure_query_cache_embedder()
            try:
                cached = await get_cache_manager().get(query, use_semantic=True)
            except Exception as e:  # noqa: BLE001 - 캐시 실패는 정상 경로로 폴백
                logger.warning(f"[RAG] [CACHE] 조회 실패 — 우회: {e}")
                cached = None
            if (
                isinstance(cached, dict)
                and isinstance(cached.get("response"), str)
                and cached[
                    "response"
                ].strip()  # 빈 문자열/공백/Nones는 히트로 취급하지 않음
                and float(cached.get("confidence", 0.0)) >= QUERY_CACHE_MIN_CONF
            ):
                response = cached["response"]
                is_cached = True
                cached_response = response
                intent = "general"  # 라우터가 generate로 단축 라우팅
                logger.info("[RAG] [PREPROCESS] 쿼리 캐시 히트 — generate 단축 경로")

    return {
        "intent": intent,
        "is_cached": is_cached,
        "cached_response": cached_response,
        "search_weights": weights,
        # 턴 시작 시 이전 턴의 재작성 쿼리 잔재 제거 (reset_or_append 리듀서의 리셋 신호)
        "search_queries": [],
        "retry_count": 0,
    }


# 최종 컨텍스트에서 유지할 섹션의 최소 길이 임계값(문자 수).
# 이 값보다 짧은 섹션은 단편 청크로 판단되어 컨텍스트에서 제외된다.
# 50자 미만이면 대부분 "단락의 부제목만 추출된" 의미 없는 조각이므로
# generate 컨텍스트 오염을 막기 위해 드롭한다 (실측: `[GENERATE] 문서 0 길이: 43`).
_MIN_CONTEXT_SECTION_LEN = 50


def _filter_min_section_len(docs: list[Document]) -> list[Document]:
    """50자 미만 초단문 섹션을 최종 컨텍스트에서 제거합니다.

    임계값: ``_MIN_CONTEXT_SECTION_LEN`` (50자). 해당 길이 이상인 섹션만
    ``kept`` 로 보관된다.

    단답형 fallback 가드 (line 276 ``return kept if kept else docs[:1]``):
    모든 입력 문서가 50자 미만이라 ``kept`` 가 비어 있는 극단 케이스에서는
    컨텍스트가 완전히 비는 것을 막기 위해 원본에서 정확히 1개 문서만 유지한다.
    이 가드는 "무조건 1개 유지"가 아니라 "kept가 비었을 때만" 동작하므로,
    하나라도 50자 이상 문서가 있으면 그 문서들만 반환된다.
    """
    if not docs:
        return []
    kept = [d for d in docs if len(d.page_content) >= _MIN_CONTEXT_SECTION_LEN]
    return kept if kept else docs[:1]


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        return {}

    t0 = time.perf_counter()
    _retrieve_op = _enter_stage(OperationType.DOCUMENT_RETRIEVAL)
    _retrieve_op.__enter__()

    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    query = get_state_attr(state, "input")
    search_queries = get_state_attr(state, "search_queries")
    if search_queries:
        query = search_queries[-1]
        logger.debug(
            f"[RAG] [RETRIEVE] 재구성된 쿼리 사용: '{query}' (Retry: {get_state_attr(state, 'retry_count')})"
        )
        SessionManager.add_status_log(
            f"Retrying search with rewritten query: {query}",
            session_id=_get_session_id(config),
        )
    else:
        logger.debug(f"[RAG] [RETRIEVE] 원본 쿼리 기반 검색 시작: '{query}'")

    cfg = config.get("configurable", {})

    if writer is not None:
        await adispatch_custom_event(
            "graph_status",
            {"status": "관련 지식 검색 중..."},
            config=config,
        )
    SessionManager.add_status_log(
        f"Searching knowledge base: {query}", session_id=_get_session_id(config)
    )

    bm25 = cfg.get("bm25_retriever")
    faiss = cfg.get("faiss_retriever")

    from common.config import DYNAMIC_TOP_K_CONFIG, ENSEMBLE_WEIGHTS, RETRIEVER_CONFIG
    from core.retriever_factory import (
        search_bm25_with_scores,
        search_faiss_with_scores,
    )

    # [수정] R3a-01: 리트리버가 실제 반환한 점수를 메타데이터에서 캡처하도록
    # 점수 주입 검색 헬퍼를 사용한다 (FAISS similarity_search_with_score / BM25 get_top_n+get_scores).
    search_k = int(RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 25))

    search_tasks = {}
    if bm25:
        search_tasks["bm25"] = asyncio.create_task(
            search_bm25_with_scores(bm25, query, search_k)
        )
    if faiss:
        search_tasks["faiss"] = asyncio.create_task(
            search_faiss_with_scores(faiss, query, search_k)
        )

    results = {}
    if search_tasks:
        task_names = list(search_tasks.keys())
        task_results = await asyncio.gather(*search_tasks.values())
        results = dict(zip(task_names, task_results, strict=False))

    search_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        f"[RAG] [RETRIEVE] 검색 결과 확보 (BM25: {len(results.get('bm25', []))}, Vector: {len(results.get('faiss', []))})"
    )

    # [수정] R3a-01: 소스별 dict {"bm25": [...], "faiss": [...]}를 그대로 전달해
    # SearchResultAggregator._rrf_fusion_2node가 가중치와 소스별 순위를 실제 적용.
    # score=0.5 하드코딩 폴백 제거 — 리트리버 점수를 명시 캡처하고, 점수가 없는
    # 경우(테스트 대역 등)에만 순위 보존 폴백(0.0)을 사용한다.
    source_results: dict[str, list[AggregatedSearchResult]] = {}
    doc_map: dict[str, Document] = {}
    for source, res in results.items():
        node_results = []
        for doc in res:
            doc_id = doc.metadata.get("doc_id", fast_hash(doc.page_content))
            raw_score = doc.metadata.get("score")
            if raw_score is None:
                logger.debug(
                    f"[RAG] [RETRIEVE] {source} 소스 문서에 score 메타데이터 없음 — 순위 보존 폴백"
                )
                raw_score = 0.0
            node_results.append(
                AggregatedSearchResult(
                    doc_id=doc_id,
                    content=doc.page_content,
                    score=float(raw_score),
                    node_id=source,
                    metadata=doc.metadata,
                )
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        if node_results:
            source_results[source] = node_results

    aggregator = SearchResultAggregator()
    weights = get_state_attr(state, "search_weights") or {
        "bm25": ENSEMBLE_WEIGHTS[0],
        "faiss": ENSEMBLE_WEIGHTS[1],
    }

    # 실제 집계에 적용되는 가중치만 로그로 기록 (단일 소스 상황 포함)
    applied_desc = ", ".join(
        f"{nid}({weights.get(nid, 1.0):.2f})" for nid in source_results
    )
    logger.debug(f"[RAG] [RETRIEVE] 하이브리드 가중치 적용: {applied_desc}")

    aggregated, _ = aggregator.aggregate_results(
        source_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=search_k,
        weights=weights,
    )

    # [수정] R3a-04: 동적 Top-K 임계값을 RRF 점수 스케일(1/(k+rank))로 보정.
    # 가중치 합산 RRF의 상위권 최대 gap은 ~0.003 수준이므로 config 값(기본 0.003)으로 판정.
    dynamic_cfg = DYNAMIC_TOP_K_CONFIG
    gap_threshold = float(dynamic_cfg.get("gap_threshold", 0.003))
    min_candidates = int(dynamic_cfg.get("min_candidates", 12))
    max_candidates = int(dynamic_cfg.get("max_candidates", 18))

    if len(aggregated) >= 10:
        top_1_score = aggregated[0].aggregated_score
        top_10_score = aggregated[9].aggregated_score
        score_gap = top_1_score - top_10_score

        # 상위 그룹이 명확하면 후보군을 축소하고, 그렇지 않으면 최대 후보를 유지
        dynamic_top_k = min_candidates if score_gap > gap_threshold else max_candidates
        logger.debug(
            f"[RAG] [RETRIEVE] Dynamic Top-K 적용: {dynamic_top_k} "
            f"(Score Gap: {score_gap:.4f} / 임계값: {gap_threshold:.4f})"
        )
    else:
        dynamic_top_k = min_candidates

    # [R3b-02] RRF 집계 점수를 doc_id 기준으로 보관 — 6자 미만 쿼리 리랭킹 생략 경로에서
    # rerank_score로 기록해 grade short-circuit이 0.0으로만 평가되는 것을 방지한다.
    rrf_scores: dict[str, float] = {}
    final_docs = []
    for r in aggregated[:dynamic_top_k]:
        doc = doc_map.get(
            r.doc_id,
            Document(page_content=r.content, metadata=r.metadata),
        )
        rrf_scores[r.doc_id] = float(r.aggregated_score)
        final_docs.append(doc)

    aggregate_ms = (time.perf_counter() - t0) * 1000

    if not final_docs:
        q_len = len(query) if query else 0
        logger.warning(
            f"[RAG] [RETRIEVE] 검색 결과가 전혀 없습니다 (Query Length: {q_len})"
        )
        SessionManager.add_status_log(
            "No documents found.", session_id=_get_session_id(config)
        )
        return {"relevant_docs": []}

    from core.async_reranker import get_async_reranker

    reranker = await get_async_reranker()
    rerank_top_k = min(int(GRADING_CONFIG.get("top_k", 5)), len(final_docs))
    if len(query or "") < 6:
        ranked_docs = final_docs[:rerank_top_k]
        # [R3b-02] 6자 미만 쿼리는 리랭킹을 생략하되 rerank_score(RRF 집계 점수)를 기록해
        # grade short-circuit이 0.0으로만 평가되는 것을 방지한다.
        for doc in ranked_docs:
            doc_key = doc.metadata.get("doc_id", fast_hash(doc.page_content))
            doc.metadata["rerank_score"] = rrf_scores.get(doc_key, 0.0)
    else:
        ranked_docs, _ = await reranker.rerank(
            final_docs,
            query=query,
            top_k=rerank_top_k,
        )
    rerank_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"[RAG] [RETRIEVE] 리랭킹 선별 완료: {len(final_docs)}개 후보 중 {len(ranked_docs)}개 최종 선별"
    )

    from typing import cast

    context_docs = cast(list[Document], ranked_docs)
    merged_context_docs = await asyncio.to_thread(
        _merge_adjacent_chunks, context_docs, max_tokens=800
    )
    # [T11] 50자 미만 초단문 섹션 드롭 — generate 컨텍스트가 단편 청크로 오염되지
    # 않도록 필터링한다. 백필 없음: 드롭으로 인한 문서 수 감소는 허용
    # (grade_top_n=3, top_k=5 하한 이상 유지).
    filtered_docs = _filter_min_section_len(merged_context_docs)
    dropped = len(merged_context_docs) - len(filtered_docs)
    if dropped > 0:
        logger.warning(
            "[RAG] [CTX] %d개 초단문 섹션 드롭 (< %d자)",
            dropped,
            _MIN_CONTEXT_SECTION_LEN,
        )
        SessionManager.add_status_log(
            f"{dropped}개 초단문 섹션(50자 미만)을 컨텍스트에서 제외했습니다.",
            session_id=_get_session_id(config),
        )
    merged_context_docs = filtered_docs
    logger.debug(
        f"[RAG] [RETRIEVE] 하이브리드 검색 및 문맥 보강 완료: 최종 {len(merged_context_docs)}개 섹션 구성"
    )

    merge_ms = (time.perf_counter() - t0) * 1000
    total_ms = merge_ms
    logger.debug(
        f"[RAG] [RETRIEVE][TIMING] search_ms={search_ms:.1f} "
        f"aggregate_ms={aggregate_ms:.1f} rerank_ms={rerank_ms:.1f} "
        f"merge_ms={merge_ms:.1f} total_ms={total_ms:.1f}"
    )

    _add_stage_ms("retrieve_ms", total_ms)
    _retrieve_op.__exit__(None, None, None)
    return {"relevant_docs": merged_context_docs}


class UnifiedGradeRewriteResponse(BaseModel):
    """문서 평가 + 쿼리 재구성을 단일 호출로 통합."""

    action: str = Field(description="'generate' 또는 'rewrite'")
    is_relevant: bool = Field(
        description="문서가 질문에 답변하기에 충분한 정보를 포함하고 있는지 여부"
    )
    relevant_entities: list[str] = Field(
        default_factory=list,
        description="질문과 관련된 문서 내 핵심 키워드나 고유 명사 목록",
    )
    reason: str = Field(description="결정에 대한 구체적인 근거")
    optimized_query: str | None = Field(
        default=None,
        description="rewrite 시에만 채울 최적화된 검색어",
    )


async def grade_documents(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """검색된 문서들의 관련성을 LLM으로 평가하고, 부적절 시 재검색 쿼리를 동시에 생성합니다."""
    # 운영자 제어 채점 단계 opt-out (GRADING_ENABLED=False): LLM 호출 없이
    # 즉시 generate로 직행. T8 메모이제이션/is_cached 단축 경로보다 우선.
    if not GRADING_ENABLED:
        logger.info("[RAG] [GRADE] 채점 단계 비활성화 — generate로 직행")
        return {"intent": "generate", "route": "generate"}

    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        # R1a-06: 라우팅은 intent가 아닌 route 채널로 결정하므로 여기서도 명시적으로 설정
        return {"route": "generate"}

    grade_start = time.perf_counter()
    _grade_op = _enter_stage(OperationType.LLM_INFERENCE)
    _grade_op.__enter__()

    retry_count = get_state_attr(state, "retry_count", 0)
    max_retries = GRADING_CONFIG.get("max_retries", 2)
    if retry_count >= max_retries:
        logger.info(
            f"[RAG] [GRADE] 최대 재시도 횟수({retry_count}/{max_retries}) 도달. 즉시 생성 단계로 이동."
        )
        _grade_op.__exit__(None, None, None)
        return {"intent": "generate", "route": "generate"}

    docs = get_state_attr(state, "relevant_docs")
    if not docs:
        logger.info("[RAG] [GRADE] 문서가 없어 즉시 재구성 단계로 이동")
        # R1a-01: 증가는 grade 단일 지점에서만. "문서 없음" 경로가 루프 종료를
        # rewrite 폴백의 간접 증가에 의존하던 취약 결합을 명시적 델타 +1로 해소.
        _grade_op.__exit__(None, None, None)
        return {
            "intent": "transform",
            "route": "transform",
            "retry_count": 1,
        }

    # Wave 3 grade reduction: 동일 세션 내 동일 (query, doc-set) 반복 질의 시
    # 이전 LLM 판단을 재사용한다 (메모 해시 miss 시에만 LLM 호출).
    # 메모 저장 실패는 채점을 깨뜨리지 않도록 LLM 경로로 폴백한다.
    sid = _get_session_id(config)
    try:
        memo_key = _grade_memo_key(state, docs)
        grade_memo = SessionManager.get(_GRADE_MEMO_KEY, session_id=sid, default={})
        if isinstance(grade_memo, dict) and memo_key in grade_memo:
            decision = grade_memo[memo_key]
            reused_route = (
                decision.get("route", "generate")
                if isinstance(decision, dict)
                else "generate"
            )
            logger.info("[RAG] [GRADE] 메모이즈된 판단 재사용")
            SessionManager.add_status_log(
                "이전 평가 결과를 재사용합니다.", session_id=sid
            )
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} memo_hit=True")
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            # [FIX] 메모 히트 시에도 재시도 카운터를 누적해야 한다.
            # 메모 재사용 경로가 retry_count 를 반환하지 않으면 reset_or_add
            # 리듀서가 카운터를 증가시키지 않아 hardcap(>= max_retries)에
            # 도달하지 못하고 무한 재귀(GraphRecursionError)가 발생한다.
            return {"intent": "generate", "route": reused_route, "retry_count": 1}
    except Exception as e:  # noqa: BLE001 - 메모 실패는 치명적이지 않음
        logger.debug(f"[RAG] [GRADE] 메모 조회 실패, LLM 경로로 진행: {e}")

    # Short-circuit: 리랭킹 점수가 충분히 높으면 LLM 검증 생략
    max_rerank_score = max(
        (d.metadata.get("rerank_score", 0.0) for d in docs), default=0.0
    )
    # [R3b-02] rerank_score 스케일은 엔진에 따라 다르다 — FlashRank 시그모이드(실측 0.06~0.91)와
    # bi-encoder 코사인(실측 0.32~0.57). 활성 엔진에 따라 임계값을 분기한다.
    from core.async_reranker import get_active_rerank_engine

    if get_active_rerank_engine() == "semantic":
        min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip_semantic", 0.60)
    else:
        min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip", 0.85)
    if max_rerank_score >= min_score_to_skip:
        logger.info(
            f"[RAG] [GRADE] Short-circuit 활성화 (Max Rerank Score: {max_rerank_score:.3f} >= {min_score_to_skip})"
        )
        SessionManager.add_status_log(
            "High-confidence knowledge found. Generating the answer now.",
            session_id=_get_session_id(config),
        )
        grade_ms = (time.perf_counter() - grade_start) * 1000
        logger.info(f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=True")
        _add_stage_ms("grade_ms", grade_ms)
        _grade_op.__exit__(None, None, None)
        return {"intent": "generate", "route": "generate"}

    query = get_state_attr(state, "input")
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import json
    import re

    if writer is not None:
        await adispatch_custom_event(
            "graph_status", {"status": "문서 관련성 검증 중..."}, config=config
        )

    test_docs = docs[: int(GRADING_CONFIG.get("grade_top_n", 3))]
    context_text = "\n\n".join(
        [f"DOC {i + 1}: {d.page_content}" for i, d in enumerate(test_docs)]
    )

    # 통합 프롬프트: 평가 + 재작성 동시 지시
    unified_prompt = (
        "당신은 문서 관련성 평가자이자 검색 쿼리 최적화 전문가입니다.\n\n"
        f"[질문]\n{query}\n\n"
        f"[검색된 문서 (상위 3개)]\n{context_text}\n\n"
        "[작업]\n"
        "1. 위 문서들이 질문에 답하기에 충분한지 평가하세요 (is_relevant: true/false)\n"
        "2. 충분하지 않다면, 더 나은 검색 결과를 위한 최적화된 쿼리를 작성하세요 (optimized_query)\n"
        "3. 판단 근거(reason)와 관련 엔티티(relevant_entities)도 포함하세요.\n\n"
        '출력은 반드시 JSON 형식이어야 합니다. (예: {"action": "generate", "is_relevant": true, '
        '"relevant_entities": ["A"], "reason": "...", "optimized_query": null})'
    )

    call_config = (
        {"configurable": {**cfg, "messages": []}}
        if cfg
        else {"configurable": {"messages": []}}
    )

    # PHASE 2: Eagerly start generate (buffered) so its LLM round-trip overlaps
    # with the grade LLM call below. Adopted by generate on the common
    # route=generate path; cancelled if grade routes to transform.
    _start_speculative_generate(state, config, writer)

    try:
        if llm is None:
            raise ValueError("LLM is not initialized")

        # JSON 모드 강제 (구조화 출력 대신) — 단일 호출로 완성
        try:
            async with ModelManager.inference_session():
                json_llm = llm.bind(response_format={"type": "json_object"})
                result = await _safe_invoke(json_llm, unified_prompt, call_config)
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)
            parsed = UnifiedGradeRewriteResponse(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[RAG] [GRADE] JSON 모드 실패, 수동 파싱 시도: {e}")
            async with ModelManager.inference_session():
                raw_res = await _safe_invoke(llm, unified_prompt, call_config)
            raw_content = (
                raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            )
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                parsed = UnifiedGradeRewriteResponse(**data)
            else:
                raise ValueError("JSON 패턴을 찾을 수 없습니다.") from None

        if parsed.action == "generate" or parsed.is_relevant:
            logger.info(f"[RAG] [GRADE] 관련성 확인: YES ({parsed.reason})")
            SessionManager.add_status_log(
                "검색된 지식의 관련성이 확인되었습니다.",
                session_id=_get_session_id(config),
            )
            _store_grade_memo(state, docs, {"route": "generate"}, sid)
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(
                f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
            )
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            # PHASE 2: route=generate → keep the warm speculative generate; the
            # real generate node adopts it (single LLM call, started earlier).
            return {"intent": "generate", "route": "generate"}
        else:
            optimized = parsed.optimized_query or query
            logger.info(
                f"[RAG] [GRADE] 관련성 확인: NO → 재작성: {optimized} ({parsed.reason})"
            )
            SessionManager.add_status_log(
                "검색 결과가 부적합하여 질문 재구성을 시도합니다.",
                session_id=_get_session_id(config),
            )
            _store_grade_memo(state, docs, {"route": "transform"}, sid)
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(
                f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
            )
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            # PHASE 2: route=transform → discard the speculative generate; its
            # buffered output must never reach the user.
            _cancel_speculative_generate(config)
            return {
                "intent": "transform",
                "route": "transform",
                "search_queries": [optimized],
                # 리듀서 reset_or_add는 합산 계약이므로 항상 상수 델타 1을 반환한다.
                # `retry_count + 1`을 반환하면 누적값이 다시 합산돼 예산이 이중 소진된다 (R1a-01).
                "retry_count": 1,
            }

    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[RAG] [GRADE] 평가 실패, 기본값(NO) 적용하여 재구성 시도: {e}")
        _store_grade_memo(state, docs, {"route": "transform"}, sid)
        grade_ms = (time.perf_counter() - grade_start) * 1000
        logger.info(
            f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
        )
        _add_stage_ms("grade_ms", grade_ms)
        _grade_op.__exit__(None, None, None)
        # PHASE 2: LLM/JSON 오류로 route=transform이 되면 speculative generate를
        # 취소·폐기해야 한다 (미노출 + 레지스트리 정리 → 이후 동일 thread_id 겹침 재활성).
        _cancel_speculative_generate(config)
        return {
            "intent": "transform",
            "route": "transform",
            "retry_count": 1,
        }


async def rewrite_query(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """grade_documents에서 이미 재작성된 쿼리를 전달합니다. (LLM 호출 없음)"""
    search_queries = get_state_attr(state, "search_queries")

    if search_queries:
        new_query = search_queries[-1]
        logger.info(f"[RAG] [REWRITE] 전달받은 재구성 쿼리 사용: '{new_query}'")
        # retry_count는 grade_documents가 이미 +1을 적용했으므로 순수 passthrough만 한다.
        # (리듀서 reset_or_add가 합산하므로 여기서 값을 내려보내면 재시도 예산이 이중 소진됨)
        return {}

    # 폴백: grade_documents에서 쿼리 생성 실패 시 원본 유지.
    # retry_count 증가는 grade_documents 단일 지점에서만 수행한다 (R1a-01).
    # 합산 리듀서(reset_or_add)에 절대 목표값을 델타로 반환하면 예산이 이중 소진된다.
    query = get_state_attr(state, "input")
    logger.info(f"[RAG] [REWRITE] 재검색 쿼리 없음, 원본 유지: '{query}'")
    return {}


def _doc_stable_id(doc: Document) -> str:
    """문서의 안정(stable) 식별자를 반환합니다.

    `format_context`, `_estimate_ctx_tokens`, verify 노드 모두 동일한 식별자를
    사용해야 하므로 단일 진원(single source of truth)으로 추출합니다.
    `doc_id` 메타데이터가 있으면 그것을, 없으면 page_content 해시를 사용합니다
    (graph_builder.py:382 / rag_core.py:69 와 동일 패턴).
    """
    return str(doc.metadata.get("doc_id", fast_hash(doc.page_content)))


# verify 노드와 테스트에서 공유하는 안정-id 인용 검증 정규식.
# 형식은 format_context가 내보내는 토큰과 동일해야 한다 ([doc:<stable_id>]).
_RE_VERIFY_DOC_CITATION = re.compile(r"\[doc:([\w-]+)\]")


def _validate_cited_doc_ids(answer: str, docs: list[Document]) -> list[str]:
    """답변 내 [doc:<stable_id>] 인용이 docs에 실제로 존재하는지 멤버십 검사.

    위치 기반 int 범위 검사(int(...) > max_idx)가 아니라, docs에서 계산한
    안정-id 집합(valid_ids)에 cited id가 속하는지 확인한다. 존재하지 않는
    (알 수 없는/환각) id만 invalid_citations로 반환한다.
    """
    if not docs:
        # docs가 없는데 인용이 남아있으면 모두 무효.
        return [m.group(1) for m in _RE_VERIFY_DOC_CITATION.finditer(answer)]

    valid_ids = {_doc_stable_id(d) for d in docs}
    invalid: list[str] = []
    for m in _RE_VERIFY_DOC_CITATION.finditer(answer):
        cited_id = m.group(1)
        if cited_id not in valid_ids:
            invalid.append(cited_id)
    return invalid


def format_context(docs: list[Document]) -> str:
    """검색된 문서들을 LLM이 읽기 좋은 형식의 문자열로 변환합니다.

    Phase 1.4: 구조화된 답변을 위한 인용 포맷
    [doc:<stable_id>] [section:X] [page:Y] [score:Z]

    [수정] 인용 토큰은 enumerate 위치(i) 대신 안정 식별자(stable id)를 사용합니다.
    위치 기반 토큰은 rerank/retry 시 문서 순서가 바뀌면 인용이 엉뚱한 청크를
    가리키는 버그를 유발하므로, doc_id(또는 content 해시)를 씁니다.
    """
    context = ""
    for d in docs:
        stable_id = _doc_stable_id(d)
        section = d.metadata.get("current_section", "일반 본문")
        page = d.metadata.get("page", "?")
        score = d.metadata.get("rerank_score", d.metadata.get("score", 0.0))
        context += f"[doc:{stable_id}] [section:{section}] [page:{page}] [score:{score:.3f}]\n{d.page_content}\n\n"
    return context


def _estimate_ctx_tokens(docs: list, query: str) -> int:
    """Estimate prompt tokens for the analysis-context template.

    Uses a SINGLE call on the assembled full-context string so the estimate
    matches the historical semantics (and is robust to call-count-based mocks
    such as a constant ``count_tokens_rough``); the value is the initial `est`
    before per-removal decrements. O(n) overall: the context is assembled
    once here and the guard loop only decrements per removal.

    The context string is assembled inline (mirroring :func:`format_context`)
    rather than by calling the module-level ``format_context``, so this helper
    does not inflate the ``format_context`` call count observed by
    ``_apply_ctx_guard`` (invoked at most twice: pre-guard and post-loop).
    """
    if not docs:
        return count_tokens_rough(
            f"{ANALYSIS_PROTOCOL}\n\n[Context]\n\n[Question]\n{query}"
        )
    context = "".join(
        f"[doc:{_doc_stable_id(d)}] "
        f"[section:{d.metadata.get('current_section', '일반 본문')}] "
        f"[page:{d.metadata.get('page', '?')}] "
        f"[score:{float(d.metadata.get('rerank_score', d.metadata.get('score', 0.0))):.3f}]"
        f"\n{d.page_content}\n\n"
        for d in docs
    )
    return count_tokens_rough(
        f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{query}"
    )


def _apply_ctx_guard(docs: list, query: str) -> tuple[list, str, int]:
    """Enforce the num_ctx token budget on the retrieval context.

    Returns ``(trimmed_docs, context_str, removed_count)``. Documents are
    dropped from the lowest ``rerank_score`` end of a descending-ranked copy
    until the estimated token cost fits the budget, while always preserving a
    minimum of 2 documents. On each removal the remaining docs are re-formatted
    and the full context string is re-counted, so ``format_context`` is invoked
    at most once per removal (1 + removals, which is bounded by the number of
    documents).
    """
    # Pre-guard context format (initial render of all docs).
    context = format_context(docs) if docs else "일상적인 대화입니다."

    if not docs:
        return docs, context, 0

    token_budget = int((OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT) * 0.85)
    est = _estimate_ctx_tokens(docs, query)

    removed = 0
    if est > token_budget:
        # rerank_score 내림차순(최상위 문서 우선) 사본에서 낮은 점수 문서부터 제거
        ranked = sorted(
            docs,
            key=lambda d: float(d.metadata.get("rerank_score", 0.0)),
            reverse=True,
        )
        while est > token_budget and len(ranked) > 2:
            ranked.pop()
            removed += 1
            context = format_context(ranked)
            est = count_tokens_rough(
                f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{query}"
            )
        docs = ranked
        logger.info(f"[RAG] [CTX] trimmed {removed} docs, est tokens={est}")
    return docs, context, removed


# R4-04: 간접 프롬프트 인젝션 패턴 — OWASP RAG Cheat Sheet §3이 스캔을 권장하는
# "SYSTEM:", "INSTRUCTION:", "ignore previous" 계열. SYSTEM/INSTRUCTION은 콜론이
# 뒤따라야만 매칭해 일반 명사("system" 등) 오탐을 줄인다.
_INJECTION_PATTERNS = re.compile(
    r"(?:SYSTEM|INSTRUCTION)\s*:|ignore\s+(?:all\s+)?previous(?:\s+instructions?)?",
    re.IGNORECASE,
)


def _split_injection_docs(
    docs: list[Document],
) -> tuple[list[Document], list[Document]]:
    """검색 청크에서 간접 프롬프트 인젝션 패턴을 스캔해 위험 청크를 분리합니다.

    검색된 원문에 "지시를 무시하고..." 같은 명령이 포함되면 LLM이 데이터를 지시로
    오인할 수 있다. 감지된 청크는 (clean, flagged)로 나누어 컨텍스트에서 제외한다.
    """
    clean: list[Document] = []
    flagged: list[Document] = []
    for d in docs:
        content = d.page_content if isinstance(d.page_content, str) else ""
        if _INJECTION_PATTERNS.search(content):
            flagged.append(d)
        else:
            clean.append(d)
    return clean, flagged


def _coerce_chunk_content(raw: Any) -> str:
    """스트리밍 청크의 content를 항상 문자열로 정규화합니다 (R4-08).

    일부 LLM 벤더(Anthropic 스타일 등)는 content를 복합 콘텐츠 리스트로 반환한다.
    리스트는 텍스트 블록만 병합하고, 그 외 타입은 str()로 폴백한다.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(raw)


async def generate(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """최종 답변을 생성합니다."""
    # PHASE 2: If grade_documents pre-started this generate speculatively, adopt
    # the already-warm task (replay its buffered events, return its result) so
    # the grade round-trip is hidden behind generation. Single LLM call, just
    # begun earlier — routing/answer quality unchanged. The speculative instance
    # itself skips adoption via its _is_speculative tag.
    if not getattr(asyncio.current_task(), "_is_speculative", False):
        adoption = _adopt_speculative_generate(config)
        if adoption is not None:
            spec_task, spec_buffer = adoption
            logger.info("[RAG] [SPEC] warm generate 채택 — speculative 결과 반환")
            for ev in spec_buffer:
                await adispatch_custom_event(ev.name, ev.data, config=ev.config)
            return await spec_task

    # [TIMING] 쿼리당 버퍼를 로컬에 캡처 (재시도로 retrieve/grade가 재실행되어도 누적 유지)
    _query_timings = dict(getattr(_stage_timing_local, "stages", {}))
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        _emit_query_timing(_query_timings)
        return {"response": "LLM not loaded"}

    if writer is not None:
        await _dispatch_event(
            "graph_status",
            {"status": "답변 설계 및 생성 중..."},
            writer=writer,
            config=config,
        )
    # R1a-08: generate 진입 상태 로그는 단일 호출당 1회만 기록한다.
    # SessionManager.add_status_log(manager.py:381-382)가 연속 동일 로그를 중복 제거하므로
    # 노드 재실행이 발생해도 타임라인에 중복이 쌓이지 않는다. (retrieve/grade 재실행 로그는
    # T1/T4 소유 노드 영역 — 라우팅 결정 시점 로그로 개선하는 것은 별도 워크스트림)
    SessionManager.add_status_log(
        "답변 논리 설계 및 생성 시작", session_id=_get_session_id(config)
    )

    docs = get_state_attr(state, "relevant_docs") or []
    logger.info(f"[RAG] [GENERATE] 관련 문서 수: {len(docs) if docs else 0}")
    if docs:
        for i, d in enumerate(docs):
            logger.info(f"[RAG] [GENERATE] 문서 {i} 길이: {len(d.page_content)}")
    no_info_msg = "제공된 문서에서 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 입력하거나 문서 내용을 확인해 주세요."
    if not docs and get_state_attr(state, "intent") != "general":
        logger.info("[RAG] [GENERATE] 관련 문서 없음 -> 사용자 안내 메시지 생성")
        if writer is not None:
            await _dispatch_event(
                "response_chunk", {"content": no_info_msg}, writer=writer, config=config
            )
        _emit_query_timing(_query_timings)
        return {"response": no_info_msg}

    # 쿼리 캐시 단축 경로: preprocess에서 is_cached=True로 라우팅된 경우.
    # LLM 스트리밍을 완전히 건너뛴다. cached path에는 relevant_docs가 없으므로
    # 바로 아래 인젝션/CTX 가드는 사실상 no-op라 안전이 보존된다.
    if get_state_attr(state, "is_cached"):
        cached_response = get_state_attr(state, "cached_response")
        if cached_response:
            logger.info("[RAG] [GENERATE] 캐시 응답 스트리밍 (LLM 미호출)")
            if writer is not None:
                await _dispatch_event(
                    "response_chunk",
                    {"content": cached_response},
                    writer=writer,
                    config=config,
                )
            _emit_query_timing(_query_timings)
            return {"response": cached_response}
        # cached_response가 비어 있으면(오염된 캐시 항목 등) 정상 생성 경로로
        # 폴백하지 않고 빈 컨텍스트로 LLM을 호출하지 않도록 무정보 안내를 반환한다.
        # (Tier 2 방어: 기존 오염 캐시가 있어도 환각 생성 차단)
        logger.warning(
            "[RAG] [GENERATE] 캐시 히트이나 응답이 비어 있음 — 무정보 안내 반환"
        )
        if writer is not None:
            await _dispatch_event(
                "response_chunk",
                {"content": no_info_msg},
                writer=writer,
                config=config,
            )
        _emit_query_timing(_query_timings)
        return {"response": no_info_msg}

    # R4-04: 검색 청크 인젝션 패턴 스캔 — 발견된 청크는 컨텍스트에서 제외하고 경고를 남긴다.
    # 격리 결과도 R1a-03과 동일한 원칙으로 노드 반환을 통해 최종 상태에 반영한다.
    state_docs: list[Document] | None = None
    if docs:
        clean_docs, flagged_docs = _split_injection_docs(docs)
        if flagged_docs:
            logger.warning(
                f"[RAG] [INJECTION] 프롬프트 인젝션 패턴 감지 → "
                f"{len(flagged_docs)}개 청크 격리 (총 {len(docs)}개 중)"
            )
            SessionManager.add_status_log(
                "검색 문서 중 프롬프트 인젝션 패턴이 감지되어 답변 생성에서 제외되었습니다.",
                session_id=_get_session_id(config),
            )
            docs = clean_docs
            state_docs = clean_docs
        if not docs:
            logger.warning(
                "[RAG] [INJECTION] 전체 검색 문서가 인젝션 패턴으로 격리됨 — 안내 메시지 반환"
            )
            if writer is not None:
                await _dispatch_event(
                    "response_chunk",
                    {"content": no_info_msg},
                    writer=writer,
                    config=config,
                )
            _emit_query_timing(_query_timings)
            return {"response": no_info_msg, "relevant_docs": []}

    # [CTX 가드] num_ctx 대비 prompt 추정 토큰이 예산을 초과하면 rerank_score가 낮은
    # 문서부터 제거하여 컨텍스트 초과(overflow)를 방지합니다. (최소 2문서 유지)
    # R4-02: num_predict(출력 예산)를 예약한 후 85%만 입력에 허용한다.
    # 입력+출력 합계가 num_ctx를 넘지 않도록 상한 = (num_ctx - num_predict) * 0.85.
    # O(n^2) 재포맷 방지를 위해 추출한 _apply_ctx_guard에서 per-doc 합산 추정 + 단일
    # 재포맷을 수행한다. (R1a-03 원칙에 따라 트림 결과는 반환 dict로 전달)
    query = get_state_attr(state, "input") or ""
    docs, context, removed = _apply_ctx_guard(docs, query)
    if removed:
        state_docs = docs

    # Phase 1.3: 구조화된 답변 출력을 위한 프롬프트 템플릿 사용
    structured_prompt = PROMPT_TEMPLATES_CONFIG.get(
        "structured_output", ANALYSIS_PROTOCOL
    )
    prompt_version = PROMPT_TEMPLATES_CONFIG.get("version", "1.0")

    # 컨텍스트와 질문을 템플릿에 주입
    query = get_state_attr(state, "input")
    # JSON 스키마의 중호({})가 format() 플레이스홀더로 해석되지 않게 보호
    # {context}와 {query}만 남기고 나머지는 이스케이프
    safe_prompt = structured_prompt.replace("{context}", "{{context}}").replace(
        "{query}", "{{query}}"
    )
    safe_prompt = safe_prompt.replace("{", "{{").replace("}", "}}")
    safe_prompt = safe_prompt.replace("{{context}}", "{context}").replace(
        "{{query}}", "{query}"
    )
    formatted_prompt = safe_prompt.format(context=context, query=query)

    sys_msg = SystemMessage(content=formatted_prompt)
    human_msg = HumanMessage(
        content="Think step by step, then output ONLY the JSON object."
    )

    full_response = ""
    full_thought = ""
    last_metadata = {}

    gen_start = time.perf_counter()
    ttft_ms: float | None = None
    _generate_op = _enter_stage(OperationType.LLM_INFERENCE)
    _generate_op.__enter__()

    # 구조화된 출력 모드인지 확인 (PROMPT_TEMPLATES_CONFIG에 structured_output이 있으면 구조화 모드)
    use_structured_output = "structured_output" in PROMPT_TEMPLATES_CONFIG

    async with ModelManager.inference_session():
        # JSON 모드 강제 바인딩(grade 노드 1053행과 동일) — qwen3:4b가 값 내부에
        # 이스케이프 없는 큰따옴표를 넣어 json.loads가 실패(Expecting ',' delimiter)하는
        # 깨진 JSON 출력을 막는다.
        json_llm = llm.bind(response_format={"type": "json_object"})
        async for chunk in json_llm.astream([sys_msg, human_msg], config=config):
            if hasattr(llm, "_convert_chunk_to_thought_and_content"):
                content_chunk, thought_chunk = (
                    llm._convert_chunk_to_thought_and_content(chunk)
                )
            else:
                # Fallback for LLMs without thought/content separation.
                # R4-08: content가 리스트(복합 콘텐츠)면 텍스트 블록을 병합해
                # str+list TypeError를 방지한다 (hasattr 분기 구조는 유지).
                content_chunk = _coerce_chunk_content(
                    chunk.content if hasattr(chunk, "content") else str(chunk)
                )
                thought_chunk = ""
            if content_chunk and ttft_ms is None:
                ttft_ms = (time.perf_counter() - gen_start) * 1000

            if thought_chunk:
                full_thought += thought_chunk
            if content_chunk:
                full_response += content_chunk
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                last_metadata = chunk.response_metadata

            # 원시 JSON 청크도 항상 UI로 전송한다.
            # 구조화 모드(raw_json=True)에서는 파싱 전 원시 토큰을 먼저 띄우고,
            # 파싱 후 final_answer로 대체되도록 한다.
            if (content_chunk or thought_chunk) and writer is not None:
                await _dispatch_event(
                    "response_chunk",
                    {
                        "content": content_chunk,
                        "thought": thought_chunk,
                        "raw_json": use_structured_output,
                    },
                    writer=writer,
                    config=config,
                )

    generate_ms = (time.perf_counter() - gen_start) * 1000
    logger.info(
        f"[RAG] [GENERATE][TIMING] generate_ms={generate_ms:.1f} "
        f"ttft_ms={(ttft_ms or 0.0):.1f} output_chars={len(full_response)}"
    )
    _query_timings["generate_total_ms"] = (
        _query_timings.get("generate_total_ms", 0.0) + generate_ms
    )
    _query_timings["ttft_ms"] = _query_timings.get("ttft_ms", 0.0) + float(
        ttft_ms or 0.0
    )
    _generate_op.__exit__(None, None, None)

    # Phase 1.2: 구조화된 답변 파싱 (JSON 출력 기대)
    parsed_answer = None
    parse_failed = False
    json_str = full_response.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    def _extract_partial(json_blob: str) -> dict[str, Any]:
        """깨진 JSON에서 final_answer/citations를 정규식으로 복구한다.

        모델이 문자열 값 내부에 이스케이프 없이 큰따옴표를 넣어 json.loads가
        실패하는 경우(Expecting ',' delimiter 등), citations[] 유실을 막기 위해
        최소한의 필드를 추출한다. 복구 불가능하면 빈 dict 반환.
        """
        recovered: dict[str, Any] = {}
        m = re.search(
            r'"final_answer"\s*:\s*"(.*?)"\s*,\s*"citations"', json_blob, re.DOTALL
        )
        if m:
            recovered["final_answer"] = m.group(1)
        else:
            m2 = re.search(r'"final_answer"\s*:\s*"(.*)', json_blob, re.DOTALL)
            if m2:
                recovered["final_answer"] = m2.group(1).rstrip('"').rstrip()
        cites = re.findall(
            r'"doc_id"\s*:\s*(\d+).*?"text_span"\s*:\s*"([^"]*)"',
            json_blob,
            re.DOTALL,
        )
        if cites:
            recovered["citations"] = [
                {
                    "doc_id": int(d),
                    "text_span": t,
                    "section": "",
                    "page": 0,
                    "score": 0.0,
                }
                for d, t in cites
            ]
        return recovered

    try:
        parsed_data = json.loads(json_str)

        # LLM 출력 필드명 매핑: thinking → reasoning (일부 모델 호환성)
        if "thinking" in parsed_data and "reasoning" not in parsed_data:
            parsed_data["reasoning"] = parsed_data.pop("thinking")
            logger.debug("[RAG] [GENERATE] 'thinking' 필드를 'reasoning'으로 매핑함")

        parsed_answer = AnswerStructure(**parsed_data)
        logger.info(
            f"[RAG] [GENERATE] 구조화된 답변 파싱 성공: prompt_version={prompt_version}"
        )

        # 구조화된 출력 모드일 때: 파싱된 final_answer를 UI로 스트리밍
        if (
            use_structured_output
            and parsed_answer
            and parsed_answer.reasoning
            and writer is not None
        ):
            await _dispatch_event(
                "response_chunk",
                {"content": "", "thought": parsed_answer.reasoning},
                writer=writer,
                config=config,
            )
        # P3: 인라인 [doc:...] 폴백과 별도로 citations[] 자체를 스트림에 태워
        # 안정적 doc_id 기반 렌더링을 가능케 한다.
        if parsed_answer and writer is not None:
            await _dispatch_event(
                "citations",
                {"citations": [c.model_dump() for c in parsed_answer.citations]},
                writer=writer,
                config=config,
            )
    except (json.JSONDecodeError, ValueError) as e:
        parse_failed = True
        # 진단: 재현/원인 분석을 위해 원시 출력을 기록 (blind spot 해소)
        logger.warning(
            f"[RAG] [GENERATE] JSON 파싱 실패, 폴백 사용: {e} "
            f"(output_chars={len(full_response)})"
        )
        logger.debug(f"[RAG] [GENERATE] 파싱 실패 원시 출력:\n{json_str}")
        # 강건화: 깨진 JSON에서 final_answer/citations 부분 복구 시도
        recovered = _extract_partial(json_str)
        if recovered:
            logger.info(
                f"[RAG] [GENERATE] 부분 복구 성공: "
                f"final_answer={'Y' if recovered.get('final_answer') else 'N'}, "
                f"citations={len(recovered.get('citations', []))}"
            )
        parsed_answer = AnswerStructure(
            reasoning=full_thought or "추론 과정 파싱 실패",
            final_answer=recovered.get("final_answer", full_response),
            citations=recovered.get("citations", []),
            confidence=0.5,
        )

    input_tokens = last_metadata.get("prompt_eval_count", 0)
    output_tokens = last_metadata.get("eval_count", 0)
    result: dict[str, Any] = {
        "response": parsed_answer.final_answer if parsed_answer else full_response,
        "thought": parsed_answer.reasoning if parsed_answer else full_thought,
        "citations": [c.model_dump() for c in parsed_answer.citations]
        if parsed_answer
        else [],
        "confidence": parsed_answer.confidence if parsed_answer else 0.5,
        "prompt_version": prompt_version,
        "parse_failed": parse_failed,
        # R1a-05: response_metadata 등 임의 객체가 섞이면 체크포인트 전체가 pickle로
        # 강등되는 경로를 차단하기 위해 순수 타입만 저장한다.
        "performance": _sanitize_channel_value(
            {
                **last_metadata,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "relevant_docs_count": len(docs),
                "ttft_ms": ttft_ms or 0.0,
                "generate_ms": generate_ms,
            }
        ),
    }
    # R1a-03: CTX 트림/인젝션 격리로 LLM이 실제로 본 문서 목록이 상태와 다르면
    # 반환을 통해 overwrite 리소스로 최종 상태에 반영한다. 변경이 없으면 건드리지
    # 않아 retrieve가 전달한 기존 상태를 보존한다.
    if state_docs is not None:
        result["relevant_docs"] = state_docs

    # 쿼리 응답 캐시 저장 (선택적 opt-in). 캐시 히트 경로가 아니라 이번에 새로
    # 생성한 경우에만 저장한다. 신뢰도가 기준 미달이면 저장하지 않는다.
    if QUERY_CACHE_ENABLED and not get_state_attr(state, "is_cached"):
        query_for_cache = get_state_attr(state, "input", "") or ""
        conf = float(result.get("confidence", 0.0))
        if query_for_cache and conf >= QUERY_CACHE_MIN_CONF:
            sid = _get_session_id(config)
            if bool(SessionManager.get("file_hash", session_id=sid, default=None)):
                await _ensure_query_cache_embedder()
                try:
                    await get_cache_manager().set(
                        query_for_cache,
                        {
                            "response": result["response"],
                            "confidence": conf,
                            "version": _QUERY_CACHE_VALUE_VERSION,
                        },
                        ttl_seconds=QUERY_CACHE_TTL,
                        use_semantic=True,
                        persist_to_disk=False,
                    )
                except Exception as e:  # noqa: BLE001 - 캐시 저장 실패는 무시
                    logger.warning(f"[RAG] [CACHE] 저장 실패 — 무시: {e}")

    _emit_query_timing(_query_timings)
    return result


def _merge_adjacent_chunks(
    docs: list[Document], max_tokens: int = 1200
) -> list[Document]:
    """같은 페이지의 연속된 청크들을 하나로 합쳐 풍부한 문맥을 제공합니다 (최적화 버전)."""
    if not docs:
        return []
    if len(docs) == 1:
        return docs

    from common.utils import count_tokens_rough

    merged_docs: list[Document] = []

    working_docs = sorted(
        docs,
        key=lambda x: (
            str(x.metadata.get("source", "")),
            int(x.metadata.get("page", 0)),
            int(x.metadata.get("chunk_index", 0)),
        ),
    )

    current_doc = Document(
        page_content=working_docs[0].page_content,
        metadata=copy.copy(working_docs[0].metadata),
    )

    current_tokens = count_tokens_rough(current_doc.page_content)

    for next_doc in working_docs[1:]:
        curr_m = current_doc.metadata
        next_m = next_doc.metadata

        is_same_context = curr_m.get("source") == next_m.get("source") and curr_m.get(
            "page"
        ) == next_m.get("page")

        is_same_section = curr_m.get("current_section") == next_m.get("current_section")

        curr_end = curr_m.get("end_index")
        next_start = next_m.get("start_index")

        if curr_end is not None and next_start is not None:
            is_actually_consecutive = abs(next_start - curr_end) <= 5
        else:
            is_actually_consecutive = (
                abs(next_m.get("chunk_index", 0) - curr_m.get("chunk_index", 0)) <= 1
            )

        next_tokens = count_tokens_rough(next_doc.page_content)

        if (
            is_same_context
            and is_actually_consecutive
            and is_same_section
            and (current_tokens + next_tokens + 10) <= max_tokens
        ):
            current_doc.page_content += "\n\n" + next_doc.page_content
            current_tokens += next_tokens + 10
            current_doc.metadata["end_index"] = next_m.get("end_index", curr_end)
            current_doc.metadata["chunk_index"] = next_m.get(
                "chunk_index", curr_m.get("chunk_index")
            )
            # [R3b-05] 병합 문서의 rerank_score는 head가 아닌 그룹 내 최대값으로 결정 —
            # grade short-circuit의 max_rerank_score와 UI 신뢰도 배지가 최고 관련 청크를 대표하도록 한다.
            curr_score = float(curr_m.get("rerank_score", 0.0) or 0.0)
            next_score = float(next_m.get("rerank_score", 0.0) or 0.0)
            current_doc.metadata["rerank_score"] = max(curr_score, next_score)
        else:
            merged_docs.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content,
                metadata=copy.copy(next_doc.metadata),
            )
            current_tokens = next_tokens

    merged_docs.append(current_doc)
    return merged_docs


# Phase 1.5: Post-Generation Verification Node
async def verify_answer(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """생성된 답변의 충실도(Faithfulness)와 인용 일관성을 검증합니다."""
    import random

    from common.config import VERIFICATION_ENABLED, VERIFICATION_SAMPLE_RATE

    # 샘플링: 프로덕션에서는 일부만 검증
    if not VERIFICATION_ENABLED or random.random() > VERIFICATION_SAMPLE_RATE:
        return {"verification_route": "end"}

    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"verification_route": "end"}

    # 검증 대상 데이터
    answer = get_state_attr(state, "response", "")
    docs = get_state_attr(state, "relevant_docs") or []
    query = get_state_attr(state, "input", "")

    # 컨텍스트 구성
    context = format_context(docs) if docs else ""

    # 인용 일관성 검사: 모든 [doc:<stable_id>]가 실제 문서 집합에 존재하는지
    # (위치 기반 int 범위 검사가 아니라 stable id 멤버십 검사).
    invalid_citations = _validate_cited_doc_ids(answer, docs)

    if invalid_citations:
        logger.warning(f"[RAG] [VERIFY] 유효하지 않은 인용 감지: {invalid_citations}")
        return {
            "verification_route": "regenerate",
            "verification_issues": [f"유효하지 않은 인용: {invalid_citations}"],
        }
    # Faithfulness 검증: LLM으로 컨텍스트 기반 답변 검증
    verify_prompt = f"""당신은 답변 검증 전문가입니다. 아래 [Context]와 [Answer]를 보고 답변이 컨텍스트에 충실한지 판단하십시오.

[Context]
{context}

[Answer]
{answer}

[Question]
{query}

다음 JSON 형식으로만 답변하십시오:
{{
  "faithful": true/false,
  "issues": ["문제점1", "문제점2"]  // faithful이 false일 때만 채움
}}

판단 기준:
1. 답변의 모든 핵심 주장이 컨텍스트에 근거하는가?
2. 컨텍스트에 없는 정보를 추측하여 답변했는가?
3. 인용된 내용이 실제 컨텍스트와 일치하는가?
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        sys_msg = SystemMessage(
            content="답변의 충실도를 엄격하게 평가하십시오. 컨텍스트에 없는 내용이 있으면 faithful: false로 판단하십시오."
        )
        human_msg = HumanMessage(content=verify_prompt)

        async with ModelManager.inference_session():
            response = await llm.ainvoke([sys_msg, human_msg], config=config)

        import json

        content = response.content if hasattr(response, "content") else str(response)
        # JSON 추출
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        result = json.loads(json_str)
        faithful = result.get("faithful", True)
        issues = result.get("issues", [])

        if not faithful or invalid_citations:
            all_issues = issues + (
                [f"유효하지 않은 인용: {invalid_citations}"]
                if invalid_citations
                else []
            )
            logger.warning(f"[RAG] [VERIFY] 검증 실패: {all_issues}")
            # 재시도 횟수 체크 (최대 1회)
            regen_count = get_state_attr(state, "regeneration_count", 0)
            if regen_count >= 1:
                logger.warning(
                    "[RAG] [VERIFY] 최대 재생성 횟수(1회) 초과, 검증 실패 상태로 종료"
                )
                return {"verification_route": "end", "verification_issues": all_issues}
            return {
                "verification_route": "regenerate",
                "verification_issues": all_issues,
                "regeneration_count": regen_count + 1,
            }

        logger.info("[RAG] [VERIFY] 검증 통과")
        return {"verification_route": "end"}

    except Exception as e:
        logger.error(f"[RAG] [VERIFY] 검증 중 오류(재생성 시도): {e}", exc_info=True)
        # fail-closed: 검증 실패=신뢰 불가 → 통과("end")로 오인하지 않고 재생성.
        # 무한루프 방지: 재생성 1회(최대) 소진 시에만 end 폴백.
        regen_count = get_state_attr(state, "regeneration_count", 0)
        if regen_count >= 1:
            logger.warning("[RAG] [VERIFY] 재생성 소진 후 검증 실패 → end 폴백")
            return {"verification_route": "end"}
        return {
            "verification_route": "regenerate",
            "verification_issues": [f"검증 실행 오류: {e}"],
        }


async def build_graph() -> Any:
    """자가 교정형 RAG 워크플로우를 구성합니다.

    [최적화] asyncio.Lock을 사용하여 동시 빌드 요청 시 이중 컴파일을 방지합니다.
    """
    logger.info("[RAG] [GRAPH] build_graph() 호출됨")
    if _graph_cache.compiled is not None:
        logger.info("[RAG] [GRAPH] 캐시된 그래프 반환")
        return _graph_cache.compiled

    lock = _graph_cache.get_lock()
    async with lock:
        # Double-check after acquiring lock (다른 세션이 이미 빌드 완료했을 수 있음)
        if _graph_cache.compiled is not None:
            logger.info("[RAG] [GRAPH] 캐시 획득 후 캐시된 그래프 반환")
            return _graph_cache.compiled

        logger.info("[RAG] [GRAPH] 그래프 재구성 시작")
        workflow = StateGraph(GraphState)

        # 노드 등록
        workflow.add_node("preprocess", preprocess)
        workflow.add_node("retrieve", retrieve_and_rerank)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("generate", generate)

        # 엣지 설정
        workflow.add_edge(START, "preprocess")

        workflow.add_conditional_edges(
            "preprocess",
            lambda s: (
                "generate" if get_state_attr(s, "intent") == "general" else "retrieve"
            ),
            {"generate": "generate", "retrieve": "retrieve"},
        )

        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            # R1a-06: 라우팅은 intent(분류 의미)가 아닌 route(전용 라우팅 채널)로 결정.
            # route 미설정 시 명시적 기본값 generate (grade가 항상 route를 반환하므로 방어적 폴백).
            lambda s: get_state_attr(s, "route", "generate"),
            {"generate": "generate", "transform": "rewrite_query"},
        )

        workflow.add_edge("rewrite_query", "retrieve")

        # Phase 1.5: 검증 노드 추가 (기능 플래그로 제어)
        from common.config import VERIFICATION_ENABLED

        if VERIFICATION_ENABLED:
            workflow.add_node("verify_answer", verify_answer)
            workflow.add_edge("generate", "verify_answer")
            # verify_answer에서 END 또는 regenerate로 라우팅
            workflow.add_conditional_edges(
                "verify_answer",
                lambda s: get_state_attr(s, "verification_route", "end"),
                {"end": END, "regenerate": "generate"},
            )
        else:
            workflow.add_edge("generate", END)

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        # R1a-05: pickle_fallback=False — msgpack 직렬화 불가 객체는 조용히 pickle로
        # 강등하지 않고 명시적 예외를 발생시킨다. 상태 채널은 _sanitize_channel_value로
        # 순수 타입만 저장되도록 위생화한다. InMemorySaver는 실험/테스트용 저장소이므로
        # 운영 전환 시 퇴거 정책이 있는 영속 체크포인터로 교체해야 한다.
        # P4: allowed_json_modules 명시 — langchain_core.messages 타입(상태에 저장될 수
        # 있는 객체형)만 reviver allowlist에 올린다. 순수 타입(int/str/float/bool/None/
        # list/dict)은 생성자 재구성이 필요 없어 allowlist 대상이 아니다.
        memory = InMemorySaver(
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=[
                    ("langchain_core", "messages", "HumanMessage"),
                    ("langchain_core", "messages", "AIMessage"),
                    ("langchain_core", "messages", "SystemMessage"),
                ],
            )
        )

        _graph_cache.compiled = workflow.compile(checkpointer=memory)
        # R1a-02/R1b-02: 세션 삭제 시 delete_graph_thread가 해당 thread를 정리할 수
        # 있도록 saver를 전역에서 참조 가능하게 노출한다.
        _graph_cache.checkpointer = memory
        return _graph_cache.compiled
