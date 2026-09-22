"""Graph 핵심 인프라 단일 진원 (Phase 2A 병합).

구 ``_graph_cache`` (컴파일된 그래프/체크포인터 프로세스 전역 캐시) +
구 ``_graph_internals`` (타이밍·JSON 복구·공용 헬퍼) +
구 ``_glue`` (세션/이벤트/투기적 generate glue)를 흡수했다.
내부 로직은 변경 없이 이동했으며, 구경로는 alias shim으로 유지한다
(``sys.modules`` 별칭 — deep patch 타깃 ``core.graph._glue.X`` 등이
실 namespace를 계속 가리키도록 보장).

순환 의존성 방지를 위해 이 모듈은 graph_builder/_generate를
절대 top-level import하지 않는다 (``_start_speculative_generate``의
``generate`` 지연 import는 유지).
"""

import asyncio
import contextvars
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import GraphState
from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    QUERY_CACHE_ENABLED,
)
from common.utils import doc_stable_id
from core.graph._speculative_gen import (
    _spec_generate_events,
    _spec_overlap_enabled,
    _spec_registry,
    _SpecEvent,
    _SpecGenerate,
)
from core.model_loader import ModelManager
from core.session import SessionManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization.backends import ObjectCache
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


# ============================================================================
# _graph_cache 섹션 — 그래프 캐시 인프라스트럭처
# (컴파일된 그래프/체크포인터의 프로세스 전역 캐시, pure move)
# ============================================================================


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


# ============================================================================
# _graph_internals 섹션 — 공용 graph 헬퍼
# (_graph_utils + _json_utils + _grading_glue 통합분, pure move)
# ============================================================================


def get_state_attr(state: Any, key: str, default: Any = None) -> Any:
    """dict와 object(GraphState) 모두에서 속성을 안전하게 가져옵니다."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


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


async def _safe_invoke(
    llm: Any, prompt: str, config: dict, model_name: str = DEFAULT_OLLAMA_MODEL
) -> Any:
    """Safely invoke an LLM with async fallback, pinning it for the call duration.

    Pins the model via the coordinator so it cannot be evicted mid-inference
    (use-after-free defect) while the underlying ainvoke is in flight.
    """
    from core.resource_manager import get_resource_manager

    coordinator = get_resource_manager()
    async with coordinator.use_llm(model_name=model_name):
        res = llm.ainvoke(prompt, config=config)
        if asyncio.iscoroutine(res):
            return await res
        return res


def _doc_stable_id(doc: Document) -> str:
    """문서의 안정(stable) 식별자를 반환합니다.

    `format_context`, `_estimate_ctx_tokens`, verify 노드 모두 동일한 식별자를
    사용해야 하므로 단일 진원(single source of truth)으로 추출합니다.
    `doc_id` 메타데이터가 있으면 그것을, 없으면 page_content 해시를 사용합니다.
    공용 구현은 `common.utils.doc_stable_id` 를 참조합니다 (R: 중복 통합).
    """
    return doc_stable_id(doc)


# ============================================================================
# _json_utils 섹션 — JSON 복구 헬퍼 (GENERATE 단계 파싱 실패 폴백, pure move)
# ============================================================================
# 모델이 값 내부에 이스케이프 없는 큰따옴표를 넣어 json.loads 가 실패하는
# 경우(Expecting ',' delimiter) 사용한다. 모든 추출은 escape 인식 스캐너로
# 수행해 D1(메타데이터 손실)/D2(text_span 절단)/D3(final_answer 절단) 를 방지.


def _extract_json_string(blob: str, start: int) -> tuple[str | None, int]:
    """시작 위치 ``start``(큰따옴표 직후)부터 닫는 큰따옴표까지 값을 읽는다.

    이스케이프된 ``\\"`` 는 건너뛴다. 스캔 종료 시 다음 읽기 시작 위치(닫는
    따옴표 바로 뒤)를 반환한다. 값을 찾지 못하면 (None, start) 를 반환한다.
    """
    i = start
    n = len(blob)
    # 키:"  값" 형태에서 콜론 뒤 공백/개행을 건너뛰고 값의 시작 큰따옴표로 이동
    while i < n and blob[i] in (" ", "\n", "\t", "\r"):
        i += 1
    if i >= n or blob[i] != '"':
        return None, start
    i += 1
    chars: list[str] = []
    while i < n:
        ch = blob[i]
        if ch == "\\":
            if i + 1 < n:
                nxt = blob[i + 1]
                if nxt == '"':
                    chars.append('"')
                    i += 2
                    continue
                if nxt == "\\":
                    chars.append("\\")
                    i += 2
                    continue
                chars.append(ch)
                i += 1
                continue
            chars.append(ch)
            i += 1
            continue
        if ch == '"':
            # 값 내부 큰따옴표(다음 문자가 공백/문자)와 닫는 따옴표(다음이
            # 구분자)를 구분: 구분자 직전만 값 종료로 본다.
            nxt = blob[i + 1] if i + 1 < n else ""
            if nxt in (":", ",", "}", "]"):
                return "".join(chars), i + 1
            chars.append('"')
            i += 1
            continue
        chars.append(ch)
        i += 1
    return None, start


def _recover_citations(blob: str) -> list[dict[str, Any]]:
    """깨진 JSON 에서 citations[] 를 객체 단위로 추출한다.

    각 citation 객체를 독립적으로 파싱하므로 doc_id/text_span 간 교차 매칭(D4)
    을 방지한다. section/page/score 도 실제 값을 보존한다(D1 해소).
    """
    citations: list[dict[str, Any]] = []
    obj_start = blob.find('"doc_id"')
    while obj_start != -1:
        obj_end = blob.find("}", obj_start)
        if obj_end == -1:
            break
        obj = blob[obj_start : obj_end + 1]
        # doc_id 는 실제로 fast_hash() 해시 문자열일 수 있으므로 따옴표 없는
        # 해시/숫자-문자 혼합 토큰도 포착한다 (스키마 int 와 실제 데이터 불일치 보정).
        doc_id_m = re.search(r'"doc_id"\s*:\s*("?)([0-9a-fA-F]+|\d+)("?)', obj)
        if not doc_id_m:
            obj_start = blob.find('"doc_id"', obj_end)
            continue
        doc_id = doc_id_m.group(2)
        span_m = re.search(r'"text_span"\s*:', obj)
        text_span: str = ""
        if span_m:
            _span, _ = _extract_json_string(obj, span_m.end())
            text_span = _span or ""
        section_m = re.search(r'"section"\s*:', obj)
        section: str = ""
        if section_m:
            _sec, _ = _extract_json_string(obj, section_m.end())
            section = _sec or ""
        page_m = re.search(r'"page"\s*:\s*(\d+)', obj)
        page = int(page_m.group(1)) if page_m else 0
        score_m = re.search(r'"score"\s*:\s*([\d.]+)', obj)
        score = float(score_m.group(1)) if score_m else 0.0
        citations.append(
            {
                "doc_id": doc_id,
                "text_span": text_span,
                "section": section,
                "page": page,
                "score": score,
            }
        )
        obj_start = blob.find('"doc_id"', obj_end)
    return citations


def _extract_partial_answer(blob: str) -> str | None:
    """깨진 JSON 에서 final_answer 값을 escape 인식 스캐너로 추출한다 (D3 해소)."""
    key_m = re.search(r'"final_answer"\s*:', blob)
    if not key_m:
        return None
    val_m = re.search(r'"(.*)', blob[key_m.end() :], re.DOTALL)
    if not val_m:
        return None
    # val_m.start() 는 blob[key_m.end():] 에서 첫 큰따옴표의 위치(0-based).
    # +1 하면 큰따옴표 바로 다음 문자가 되어 _extract_json_string 이 값을
    # 건너뛰므로, 값 시작 큰따옴표 위치 그대로 전달한다.
    val_start = key_m.end() + val_m.start()
    value, _ = _extract_json_string(blob, val_start)
    return value


def _repair_json(blob: str) -> str | None:
    """깨진 JSON 에서 값 내부 이스케이프 없는 큰따옴표를 보정한다.

    모델이 문자열 값 안에 raw 큰따옴표를 넣어 json.loads 가 실패하는 경우,
    값의 끝은 "닫는 큰따옴표 바로 다음 문자가 구분자(``,`` ``}`` ``]``
    공백/개행)인 지점"으로 추정해 내부 큰따옴표를 ``\\"`` 로 이스케이프한다.
    복구 후 json.loads 가 성공하면 보정본을, 실패하면 None 을 반환(그때만
    regex 폴백으로 진행). 의존성 없음.
    """
    i = 0
    n = len(blob)
    out: list[str] = []
    while i < n:
        ch = blob[i]
        if ch == "{":
            out.append(ch)
            i += 1
            continue
        if ch == "}":
            out.append(ch)
            i += 1
            continue
        # "doc_id": 뒤에 따옴표 없는 해시/식별자 토큰이 오는 경우 보정.
        # 실제 doc_id 는 fast_hash() 해시 문자열인데, 모델이 스키마(integer)를
        # 따르려다 따옴표 없이 넣으면 json.loads 가 실패한다. 토큰을 "..." 로 감싼다.
        if ch == '"' and blob[i : i + 8] == '"doc_id"':
            out.append('"doc_id"')
            i += 8
            # 콜론 + 공백 건너뛰기(콜론은 아래서 다시 붙인다)
            while i < n and blob[i] in (":", " ", "\n", "\t", "\r"):
                i += 1
            if i < n and blob[i] != '"':
                # 따옴표 없는 토큰(해시/숫자-문자 혼합) -> "key": "tok" 형태로 보정
                tok_start = i
                while i < n and blob[i] not in (",", "}", "]", "\n", " ", "\t", "\r"):
                    i += 1
                tok = blob[tok_start:i]
                out.append(':"' + tok + '"')
            else:
                # 이미 따옴표 있는 정상 형태면 ":" 만 붙이고 값은 기존 스캐너가 처리
                out.append(":")
                if i < n and blob[i] == '"':
                    out.append('"')
            continue
        if ch == '"':
            # 문자열 값 시작
            out.append(ch)
            i += 1
            while i < n:
                c = blob[i]
                if c == "\\":
                    out.append(c)
                    if i + 1 < n:
                        out.append(blob[i + 1])
                        i += 2
                    else:
                        i += 1
                    continue
                if c == '"':
                    # 닫는 따옴표 추정: 키명 뒤(":" 직전) 또는 값 끝("," / "}" / "]" 직전).
                    # 그 외(공백/문자 뒤)는 값 내부 큰따옴표로 보고 이스케이프한다.
                    nxt = blob[i + 1] if i + 1 < n else ""
                    if nxt in (":", ",", "}", "]"):
                        out.append(c)
                        i += 1
                        break
                    out.append('\\"')
                    i += 1
                    continue
                # 값 내부 raw 제어문자 이스케이프 (JSON 문자열 값은 \n/\r/\t 로 표기해야 함).
                # 모델이 실제 개행/탭을 그대로 넣으면 json.loads 가 실패하므로 보정한다.
                if c == "\n":
                    out.append("\\n")
                    i += 1
                    continue
                if c == "\r":
                    out.append("\\r")
                    i += 1
                    continue
                if c == "\t":
                    out.append("\\t")
                    i += 1
                    continue
                out.append(c)
                i += 1
            continue
        out.append(ch)
        i += 1
    repaired = "".join(out)
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        return None


def _strip_json_fence(raw: str) -> str:
    """LLM 응답에서 ```json / ``` 코드 펜스를 제거합니다. (R: Group 8)

    구조화(1.2)와 verify 노드에서 동일 스트리핑을 중복하던 것을 단일 헬퍼로 통합.
    """
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ============================================================================
# _grading_glue 섹션 — 스테이지 타이밍 버퍼 (pure move)
# ============================================================================
# The buffer is a mutable dict captured into a local at generate entry so
# retries (which re-run retrieve/grade) accumulate correctly.  A speculative
# generate task created via ``asyncio.ensure_future`` inherits a shallow copy
# of the context that references the SAME dict, so accumulation across the
# main and speculative tasks is preserved while different queries (different
# tasks) keep distinct dicts.

_stage_timing_var: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar(
    "_stage_timing_var"
)


def _reset_stage_timings() -> None:
    """Clear per-query stage timing buffer (called at preprocess start)."""
    _stage_timing_var.set(
        {
            "preprocess_ms": 0.0,
            "retrieve_ms": 0.0,
            "grade_ms": 0.0,
            "generate_total_ms": 0.0,
            "ttft_ms": 0.0,
        }
    )


def _add_stage_ms(stage: str, ms: float) -> None:
    """Accumulate a stage duration into the per-query buffer."""
    stages = _stage_timing_var.get(None)
    if stages is None:
        _reset_stage_timings()
        stages = _stage_timing_var.get()
    stages[stage] = stages.get(stage, 0.0) + float(ms)


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


# ============================================================================
# _glue 섹션 — RAG 그래프 파이프라인 공용 glue 헬퍼 (pure move)
# ============================================================================
# import 방향은 strictly: graph_builder → _graph_core (no cycle).
# _generate 지연 import(함수 내부)는 순환 회피를 위해 유지한다.


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
        stale = _spec_registry.pop(thread_id, None)
        if stale is not None and stale.adopter is None:
            stale.task.cancel()
            stale.buffer.clear()
            logger.warning(
                "[RAG] [SPEC] stale orphan cancelled (thread_id=%s)", thread_id
            )
    # fall through to register a fresh speculative task

    _spec_generate_events.set([])
    from core.graph._generate import generate

    task = asyncio.ensure_future(generate(state, config, writer=writer))
    # Tag the task so the speculative generate instance does NOT adopt itself
    # (it must run normally and buffer its events instead).
    task._is_speculative = True  # type: ignore[attr-defined]
    _spec_registry[thread_id] = _SpecGenerate(task=task, buffer=[])
    # The speculative task buffers into the ContextVar list we just captured.
    _spec_registry[thread_id].buffer = _spec_generate_events.get([])
    logger.info("[RAG] [SPEC] eager generate 시작 (route 결정 대기)")
    return thread_id
