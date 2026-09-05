"""FIX-2/FIX-3/FIX-8: LangGraph 채널 경유 상태 전파 통합 테스트.

설계상의 문제 (2026-09-05 실측):
- `preprocess` 가 반환하는 `short_query`(<5자)와 `cached_response` 는
  `GraphState` 스키마에 선언돼 있지 않아 LangGraph 1.6.0 이 **조용히 드롭**한다.
  그 결과:
  * `grade_documents` 의 short_query fast path (토큰 실존 검사로 LLM grade
    생략) 가 운영에서 절대 발동하지 않았다 (`grade_ms=4047.2 short_circuit=False`).
  * `generate` 의 캐시 단축 경로가 `cached_response` 를 못 읽어 캐시 히트 시
    무정보 안내를 반환한다 (latent — `query_cache.enabled: false`).

핵심 규칙 (플랜 §3 참조):
- patch → `invalidate_graph_cache()` → `build_graph()` 순서 (빌드 시 함수 참조 캡처)
- 노드 함수는 실제 async 함수로 패치 (Mock 은 LangGraph 시그니처 introspection 실패)
- rerank_score < 0.85 로 유지 → 리랭크 short-circuit(:741) 이 fast path 검증을
  마스킹하지 않도록 한다
- `get_resource_manager` stub → `use_llm` 의 실제 모델 로드 (9.8s 지연) 차단
- unique thread_id + teardown `delete_graph_thread` (스테일 checkpoint 방지)
- grade 미호출 검증: `llm.bind(...)` 호출 횟수 추적 (grade/generate 모두 bind 사용)
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

BASE_DIR = Path(__file__).parent.parent.parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from langchain_core.documents import Document

import core.graph.graph_builder as gb  # noqa: E402
from core.graph._graph_cache import delete_graph_thread, invalidate_graph_cache

# ---------------------------------------------------------------------------
# 헬퍼: 그래프 노드 캡처 함정을 회피하는 재빌드
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_process_globals():
    """테스트 간 process-global 싱글톤 오염 방지 (Momus-4).

    - string-path patch 는 다른 테스트가 ``sys.modules`` 를 교체해 새
      SessionManager 클래스 객체를 만들면 graph_builder 의 실제 참조와
      불일치한다. 캐시/세션 상태를 각 테스트 전후로 초기화해 순서 의존
      실패를 차단한다.
    """
    invalidate_graph_cache()
    gb.SessionManager.reset()
    gb.SessionManager.set_ui_sync(None)
    yield
    invalidate_graph_cache()
    gb.SessionManager.reset()
    gb.SessionManager.set_ui_sync(None)


def _fresh_graph():
    """이전 컴파일 캐시를 무효화하고 새 그래프를 빌드한다."""
    invalidate_graph_cache()
    return asyncio.run(gb.build_graph())


class _StubCoordinator:
    """ResourceManager 대역 — use_llm/inference_session 을 no-op으로 흉내낸다.

    generate(:1259) 와 grade 의 `_safe_invoke`(:814) 는 `coordinator.use_llm(...)`
    으로 실제 모델을 로드/pin 하므로, 이 대역으로 교체해 9.8s 지연을 제거한다.
    """

    @asynccontextmanager
    async def use_llm(self, model_name, **kwargs):  # noqa: ARG002
        yield None

    @asynccontextmanager
    async def inference_session(self, timeout=None):  # noqa: ARG002
        yield


class TrackingLLM:
    """bind() 호출 횟수를 추적하는 페이크 LLM.

    grade_documents 와 generate 모두 `llm.bind(...)` 후 `.astream(...)` 을 사용.
    fast path 가 정상 동작하면 grade 의 bind 호출은 0회여야 하고 generate 만 1회.
    """

    def __init__(self, chunks=None):
        self.bind_count = 0
        self._chunks = chunks or [
            '{"reasoning": "CM3 is a multimodal LM.", '
            '"final_answer": "테스트 응답입니다.", "citations": []}'
        ]

    def bind(self, **kwargs):  # noqa: ARG002
        self.bind_count += 1
        return self

    async def ainvoke(self, *args, **kwargs):  # noqa: ARG002
        # grade 의 _safe_invoke 경로(레드 시 grade 가 발동하는 경우)용.
        return type("Res", (), {"content": ""})()

    async def astream(self, messages, config=None):  # noqa: ARG002
        for c in self._chunks:
            yield type("Chunk", (), {"content": c})()


def _base_config(llm, thread_id="test-thread"):
    return {
        "configurable": {
            "llm": llm,
            "thread_id": thread_id,
            "session_id": f"sid-{thread_id}",
            "bm25_retriever": MagicMock(),
            "faiss_retriever": MagicMock(),
        }
    }


async def _collect_graph_events(graph, config, query="cm3"):
    events = []
    async for ev in graph.astream_events({"input": query}, config=config, version="v2"):
        events.append(ev)
    return events


def _custom_events(events, name: str) -> list[dict]:
    """이름으로 커스텀 이벤트를 찾는다 (on_custom_event 의 name 은 top-level)."""
    return [
        ev["data"]
        for ev in events
        if ev.get("event") == "on_custom_event" and ev.get("name") == name
    ]


def _response_chunks(events) -> list[str]:
    """response_chunk 이벤트에서 콘텐츠를 모은다."""
    return [
        d.get("content", "")
        for d in _custom_events(events, "response_chunk")
        if d.get("content")
    ]


def _make_fake_retrieve(docs):
    """docs 를 반환하는 실제 async retrieve 노드 함수 (sig: state, config, *, writer)."""

    async def _fake_retrieve(state, config, *, writer=None):  # noqa: ARG001
        return {"relevant_docs": list(docs)}

    return _fake_retrieve


# ---------------------------------------------------------------------------
# FIX-2: Scenario A — short_query fast path (LLM grade 생략 + generate 직행)
# ---------------------------------------------------------------------------


class TestShortQueryFastPath:
    def test_fast_path_skips_llm_grade(self):
        """cm3 토큰이 상위 문서에 존재 → LLM grade 미호출 + 응답 생성.

        수정 전(채널 드롭): short_query 가 스키마에서 드롭돼 grade 의 fast
        path 가 발동하지 않으므로 grade 의 LLM bind 호출이 발생 → bind_count=2
        (grade+generate) → 이 테스트 red.
        수정 후: grade bind 없이 generate 만 호출 (bind_count==1) → green.
        """
        canned_docs = [
            Document(
                page_content=(
                    "CM3 is a causally masked multimodal model. CM3 extends "
                    "this work by modeling full document structure."
                ),
                # rerank_score < 0.85 → 리랭크 short-circuit 이 fast path 를
                # 마스킹하지 않도록 낮춘다 (fast path 만이 유일한 단축 경로).
                metadata={"rerank_score": 0.5},
            ),
            Document(
                page_content=(
                    "CM3 is not only a cross-modal model but also a standalone "
                    "language model."
                ),
                metadata={"rerank_score": 0.4},
            ),
            Document(
                page_content=(
                    "For model architecture we use the same architecture for CM3."
                ),
                metadata={"rerank_score": 0.3},
            ),
        ]

        llm = TrackingLLM()
        thread_id = "fix2-fastpath-thread"
        try:
            with (
                patch(
                    "core.graph.graph_builder.retrieve_and_rerank",
                    new=_make_fake_retrieve(canned_docs),
                ),
                patch(
                    "core.graph.graph_builder.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                # _safe_invoke 는 함수 내부에서 core.resource_manager 를 import
                patch(
                    "core.resource_manager.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
            ):
                graph = _fresh_graph()  # patch 후 재빌드 (캡처 함정 회피)
                config = _base_config(llm, thread_id)
                events = asyncio.run(_collect_graph_events(graph, config))

            # fast path 발동: grade 의 LLM bind 가 없어야 한다.
            # generate 만 bind(count==1) 해야 정상. count>=2 는 grade 가
            # 실행됐다는 뜻 (채널 드롭으로 fast path 미발동) → red.
            assert llm.bind_count == 1, (
                f"fast path 가 발동하지 않아 LLM grade 가 실행됨 (bind={llm.bind_count}회)"
            )
            chunks = _response_chunks(events)
            assert chunks, "응답 청크가 스트리밍되지 않았습니다"
            assert "테스트 응답" in "".join(chunks)
            assert "찾을 수 없습니다" not in "".join(chunks), (
                "무정보 안내 메시지 반환 (fast path 실패)"
            )
        finally:
            delete_graph_thread(thread_id)


# ---------------------------------------------------------------------------
# FIX-3: Scenario B — 쿼리 캐시 히트 → cached_response 전파
# ---------------------------------------------------------------------------


class TestQueryCachePropagation:
    def test_cached_response_streams_when_cache_hit(self):
        """캐시 히트 시 generate 가 cached_response 를 스트리밍한다.

        수정 전(채널 드롭): preprocess 는 cached_response 를 반환하지만
        스키마에 없어 드롭 → generate 의 `cached_response` 읽기가 None →
        무정보 안내(빈 응답 오염 폴백) 반환 → "CACHED" 가 스트리밍되지 않음
        → 이 테스트 red.
        수정 후: "CACHED" 스트리밍 → green.
        """
        llm = TrackingLLM()
        thread_id = "fix3-cacheprop-thread"
        try:
            # preprocess :306-328 캐시 히트 경로 stub (value-import 패치 대상)
            cm = MagicMock()
            cm.get = AsyncMock(return_value={"response": "CACHED", "confidence": 0.9})
            with (
                patch(
                    "core.graph.graph_builder.retrieve_and_rerank",
                    new=_make_fake_retrieve([]),
                ),
                patch(
                    "core.graph.graph_builder.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                patch(
                    "core.resource_manager.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                # value import — graph_builder.QUERY_CACHE_ENABLED 를 패치
                patch("core.graph.graph_builder.QUERY_CACHE_ENABLED", True),
                # _ensure_query_cache_embedder: no-op (캐시 조회 전 임베더 준비)
                patch(
                    "core.graph.graph_builder._ensure_query_cache_embedder",
                    new=AsyncMock(),
                ),
                # get_cache_manager(): 팩토리 대역 — preprocess 는
                # `await get_cache_manager().get(...)` 호출
                patch(
                    "core.graph.graph_builder.get_cache_manager",
                    return_value=cm,
                ),
                # SessionManager.get("file_hash") truthy → has_doc=True
                # 그래프가 사용하는 참조(gb.SessionManager)를 직접 패치한다.
                # 문자열 경로 patch 는 sys.modules 교체 시 graph_builder 의
                # 실제 참조와 다른 클래스 객체를 잡을 수 있어 적용되지 않는다.
                patch.object(gb.SessionManager, "get", return_value="fake-file-hash"),
            ):
                graph = _fresh_graph()
                config = _base_config(llm, thread_id)
                events = asyncio.run(_collect_graph_events(graph, config))

            chunks = _response_chunks(events)
            assert "CACHED" in "".join(chunks), (
                "cached_response 가 전파되지 않아 캐시 응답이 스트리밍되지 않았습니다."
            )
            assert "찾을 수 없습니다" not in "".join(chunks), (
                "cached_response 드롭으로 무정보 안내가 반환됐습니다 (캐시 오염 폴백)."
            )
        finally:
            delete_graph_thread(thread_id)

    def test_cache_hit_with_empty_response_falls_back_to_no_info(self):
        """오염 방어: is_cached=True + 빈 cached_response → 무정보 안내, not CACHED.

        빈 문자열 폴백 가드(:1158-1163)가 유지되는지 확인 (수정 후에도 동작).
        """
        llm = TrackingLLM()
        thread_id = "fix3-cachepollute-thread"
        try:
            cm = MagicMock()
            cm.get = AsyncMock(return_value={"response": "  ", "confidence": 0.9})
            with (
                patch(
                    "core.graph.graph_builder.retrieve_and_rerank",
                    new=_make_fake_retrieve([]),
                ),
                patch(
                    "core.graph.graph_builder.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                patch(
                    "core.resource_manager.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                patch("core.graph.graph_builder.QUERY_CACHE_ENABLED", True),
                patch(
                    "core.graph.graph_builder._ensure_query_cache_embedder",
                    new=AsyncMock(),
                ),
                patch("core.graph.graph_builder.get_cache_manager", return_value=cm),
                patch.object(gb.SessionManager, "get", return_value="fake-file-hash"),
            ):
                graph = _fresh_graph()
                config = _base_config(llm, thread_id)
                events = asyncio.run(_collect_graph_events(graph, config))

            chunks = _response_chunks(events)
            assert chunks, "응답 청크가 없습니다"
            joined = "".join(chunks)
            assert "찾을 수 없습니다" in joined, (
                "빈 cached_response 에 대한 오염 방어(무정보 안내)가 동작하지 않았습니다."
            )
            assert "CACHED" not in joined
        finally:
            delete_graph_thread(thread_id)


# ---------------------------------------------------------------------------
# FIX-8: Scenario C — 부정(transform) 분기 (토큰 부재 → 재검색 → 예산 소진)
# ---------------------------------------------------------------------------


class TestTransformBranch:
    def test_token_absent_short_query_reroutes_through_transform(self):
        """cm3 토큰이 상위 문서에 없음 → transform 분기 (재검색 루프).

        결정성: patch 된 retrieve 는 재진입마다 동일한 무관 문서를 반환하므로
        retry_count 가 reset_or_add(delta-1) 로 max_retries 까지 확정 누적.
        이후 generate 라우팅으로 자연 종료 (재귀 오류 없음).
        """
        unrelated_docs = [
            Document(
                page_content="Machine learning research explores scaling laws.",
                metadata={"rerank_score": 0.5},
            ),
            Document(
                page_content="Training large models requires massive data.",
                metadata={"rerank_score": 0.4},
            ),
            Document(
                page_content="Evaluation follows zero-shot protocols.",
                metadata={"rerank_score": 0.3},
            ),
        ]

        llm = TrackingLLM(chunks=["무관 문서 기반 응답"])
        thread_id = "fix8-transform-thread"
        try:
            with (
                patch(
                    "core.graph.graph_builder.retrieve_and_rerank",
                    new=_make_fake_retrieve(unrelated_docs),
                ),
                patch(
                    "core.graph.graph_builder.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                patch(
                    "core.resource_manager.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
            ):
                graph = _fresh_graph()
                config = _base_config(llm, thread_id)
                events = asyncio.run(_collect_graph_events(graph, config))

            # fast path 기반 transform 증명: LLM grade 무개입 (generate 만 bind).
            # 수정 전(채널 드롭): short_query=False → fast path 미발동 → LLM grade
            # 발동(bind 1회) + generate(bind 1회) = 2 → red.
            # 수정 후: fast path 가 즉시 transform 으로 보내므로 grade bind 없음 → 1.
            assert llm.bind_count == 1, (
                f"fast path 기반 transform 이 아닌 LLM grade 경로가 사용됨 (bind={llm.bind_count}회)"
            )
            chunks = _response_chunks(events)
            assert chunks, "transform 분기 후 generate 응답이 없습니다"
        finally:
            delete_graph_thread(thread_id)

    def test_no_graph_recursion_error_on_repeated_misses(self):
        """동일 무관 문서가 재진입 시 반복돼도 GraphRecursionError 없이 종료."""
        unrelated_docs = [
            Document(page_content="Irrelevant A", metadata={"rerank_score": 0.5}),
            Document(page_content="Irrelevant B", metadata={"rerank_score": 0.4}),
            Document(page_content="Irrelevant C", metadata={"rerank_score": 0.3}),
        ]

        llm = TrackingLLM(chunks=["fallback"])
        thread_id = "fix8-recursion-thread"
        try:
            with (
                patch(
                    "core.graph.graph_builder.retrieve_and_rerank",
                    new=_make_fake_retrieve(unrelated_docs),
                ),
                patch(
                    "core.graph.graph_builder.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
                patch(
                    "core.resource_manager.get_resource_manager",
                    return_value=_StubCoordinator(),
                ),
            ):
                graph = _fresh_graph()
                config = _base_config(llm, thread_id)
                events = asyncio.run(_collect_graph_events(graph, config))

            # fast path 기반 transform (grade 무개입) + 재귀 오류 없이 완결
            assert llm.bind_count == 1, (
                f"fast path 기반 transform 이 아님 (bind={llm.bind_count}회)"
            )
            assert any(ev.get("event") == "on_custom_event" for ev in events), (
                "이벤트 스트림이 불완전합니다 (재귀/패닉 의심)"
            )
        finally:
            delete_graph_thread(thread_id)
