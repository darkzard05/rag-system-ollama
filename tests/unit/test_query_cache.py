"""
단위 테스트: 쿼리 캐시 배선 (T5/T6) 검증.

대상:
- core.graph_builder.preprocess() : 세맨틱 쿼리 캐시 조회 wire
- core.graph_builder.generate()   : is_cached 단축 경로 (LLM 미호출)
- core.pipeline_builder.PipelineBuilder._invalidate_caches_on_index() : 재인덱싱 무효화

모든 테스트는 격리된 mock seam 을 사용하며 실제 Ollama/네트워크 호출은 없다.
결정론적 검증을 위해 get_cache_manager / ModelManager / adispatch_custom_event /
SessionManager 를 모두 패치한다.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph_builder import generate, preprocess
from core.pipeline_builder import GRADE_MEMO_KEY, PipelineBuilder
from core.session import SessionManager
from services.optimization.caching_optimizer import get_cache_manager


# ----------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------
def _base_config(session_id: str = "test-session-1") -> dict:
    """LangGraph RunnableConfig 형태의 실행 설정."""
    return {"configurable": {"session_id": session_id, "thread_id": session_id}}


def _base_state(query: str = "what is RAG?", **overrides: object) -> dict:
    """GraphState 필드에 맞는 상태 dict 생성."""
    state = {
        "input": query,
        "intent": None,
        "route": "generate",
        "search_queries": [],
        "relevant_docs": [],
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "is_cached": False,
        "cached_response": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


class _FakeSemanticCache:
    """캐시 무효화(.clear) 호출을 기록하는 페이크 세맨틱 캐시."""

    def __init__(self) -> None:
        self.clear_calls = 0

    async def clear(self) -> None:
        self.clear_calls += 1


def _make_cache_manager(get_return: object) -> MagicMock:
    """get()이 get_return 을 반환하고, semantic_cache 를 가진 CacheManager 페이크."""
    cm = MagicMock()
    cm.get = AsyncMock(return_value=get_return)
    cm.semantic_cache = _FakeSemanticCache()
    return cm


@pytest.fixture
def patch_cache_manager():
    """get_cache_manager() 가 지정한 페이크를 반환하도록 전역 패치."""
    with patch(
        "core.graph_builder.get_cache_manager",
        side_effect=lambda *a, **k: get_cache_manager(),
    ) as p:
        yield p


# ----------------------------------------------------------------------------
# (a) cache disabled -> miss
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_preprocess_cache_disabled_is_miss(monkeypatch):
    """QUERY_CACHE_ENABLED=False 이면 항상 is_cached=False, cached_response=None."""
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_ENABLED", False)
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_MIN_CONF", 0.85)

    # get() 이 절대 호출되지 않아야 하므로 실패하는 페이크로 교체해도 안전.
    fake_cm = _make_cache_manager({"response": "cached answer", "confidence": 0.99})

    state = _base_state("reusable query")
    with patch("core.graph_builder.get_cache_manager", return_value=fake_cm):
        result = await preprocess(state, _base_config(), writer=None)

    assert result["is_cached"] is False
    assert result["cached_response"] is None
    fake_cm.get.assert_not_called()


# ----------------------------------------------------------------------------
# (b) cache hit -> zero LLM calls, cached chunk streamed
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_cache_hit_zero_llm_calls(
    monkeypatch,
):
    """캐시 히트 시 generate 는 LLM.astream/ainvoke 를 호출하지 않고 cached 를 스트리밍."""
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_MIN_CONF", 0.85)

    cached_value = {"response": "cached answer", "confidence": 0.99}
    fake_cm = _make_cache_manager(cached_value)

    captured_events: list[dict] = []
    llm_astream_calls: list[int] = []
    llm_ainvoke_calls: list[int] = []

    llm = MagicMock()
    llm.bind.return_value = llm

    async def fake_astream(*args, **kwargs):  # pragma: no cover - must not run
        llm_astream_calls.append(1)
        raise AssertionError("llm.astream must NOT be called on cache hit")

    async def fake_ainvoke(*args, **kwargs):  # pragma: no cover - must not run
        llm_ainvoke_calls.append(1)
        raise AssertionError("llm.ainvoke must NOT be called on cache hit")

    llm.astream = fake_astream
    llm.ainvoke = fake_ainvoke

    state = _base_state(
        "reusable query",
        is_cached=True,
        cached_response="cached answer",
        intent="general",
        relevant_docs=[],
    )
    config = _base_config()
    config["configurable"]["llm"] = llm

    with (
        patch("core.graph_builder.get_cache_manager", return_value=fake_cm),
        patch(
            "core.graph_builder.adispatch_custom_event",
            side_effect=lambda name, data, config=None: captured_events.append(
                {"name": name, "data": dict(data)}
            ),
        ),
    ):
        # writer 가 None 이면 cached chunk 가 adispatch_custom_event 로 스트리밍되지 않으므로
        # 더미 writer 를 주입해 실제 배선 경로를 검증한다.
        result = await generate(state, config, writer=MagicMock())

    # (b) assertions
    assert result["response"] == "cached answer"
    assert len(llm_astream_calls) == 0
    assert len(llm_ainvoke_calls) == 0

    # 캐시 응답 청크가 스트리밍되었는지 확인
    chunk_events = [e for e in captured_events if e["name"] == "response_chunk"]
    assert chunk_events, "cached response must be streamed as response_chunk"
    assert chunk_events[0]["data"]["content"] == "cached answer"


@pytest.mark.asyncio
async def test_preprocess_cache_hit_marks_cached(monkeypatch):
    """preprocess 가 캐시 히트를 감지해 is_cached=True, intent=general 로 라우팅."""
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_MIN_CONF", 0.85)

    cached_value = {"response": "cached answer", "confidence": 0.99}
    fake_cm = _make_cache_manager(cached_value)

    state = _base_state("reusable query")
    with (
        patch("core.graph_builder.get_cache_manager", return_value=fake_cm),
        patch.object(
            SessionManager,
            "get",
            return_value="somefilehash",
        ),
        patch("core.graph_builder._ensure_query_cache_embedder", new=AsyncMock()),
    ):
        result = await preprocess(state, _base_config(), writer=None)

    assert result["is_cached"] is True
    assert result["cached_response"] == "cached answer"
    assert result["intent"] == "general"


# ----------------------------------------------------------------------------
# (c) low-confidence miss
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_preprocess_low_confidence_is_miss(monkeypatch):
    """confidence 0.5 (< QUERY_CACHE_MIN_CONF 0.85) 이면 is_cached=False."""
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_MIN_CONF", 0.85)

    cached_value = {"response": "low conf answer", "confidence": 0.5}
    fake_cm = _make_cache_manager(cached_value)

    state = _base_state("reusable query")
    with (
        patch("core.graph_builder.get_cache_manager", return_value=fake_cm),
        patch.object(
            SessionManager,
            "get",
            return_value="somefilehash",
        ),
        patch("core.graph_builder._ensure_query_cache_embedder", new=AsyncMock()),
    ):
        result = await preprocess(state, _base_config(), writer=None)

    assert result["is_cached"] is False
    assert result["cached_response"] is None


# ----------------------------------------------------------------------------
# (d) reindex invalidation
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalidate_caches_on_index_clears(monkeypatch):
    """신규 인덱싱 시 semantic_cache.clear() 와 SessionManager.delete(grade memo) 호출."""
    fake_cm = _make_cache_manager(None)
    fake_semantic = fake_cm.semantic_cache  # type: ignore[attr-defined]

    builder = PipelineBuilder(session_id="sess-x")

    deleted_keys: list[tuple] = []

    with (
        patch("core.pipeline_builder.get_cache_manager", return_value=fake_cm),
        patch.object(
            SessionManager,
            "delete",
            side_effect=lambda key, session_id=None: deleted_keys.append(
                (key, session_id)
            ),
        ),
    ):
        await builder._invalidate_caches_on_index()

    assert fake_semantic.clear_calls == 1
    assert (GRADE_MEMO_KEY, "sess-x") in deleted_keys


# ----------------------------------------------------------------------------
# (e) general intent never cached -> LLM still invoked
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_general_intent_invokes_llm(monkeypatch):
    """is_cached=False 인 일반 인사(general) 경로는 LLM.astream 을 호출한다."""
    monkeypatch.setattr("core.graph_builder.QUERY_CACHE_ENABLED", True)

    llm = MagicMock()
    llm.bind.return_value = llm
    llm_astream_calls: list[int] = []

    async def fake_astream(*args, **kwargs):
        llm_astream_calls.append(1)
        chunk = SimpleNamespace(content="hello there", response_metadata={})
        yield chunk

    llm.astream = fake_astream
    # CustomOllama 전처리 메서드 모사 (hasattr 자동 생성 방지용 명시 정의)
    llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, None)

    captured_events: list[dict] = []

    state = _base_state(
        "hi",
        intent="general",
        is_cached=False,
        cached_response=None,
        relevant_docs=[],
    )
    config = _base_config()
    config["configurable"]["llm"] = llm

    with patch(
        "core.graph_builder.adispatch_custom_event",
        side_effect=lambda name, data, config=None: captured_events.append(
            {"name": name, "data": dict(data)}
        ),
    ):
        result = await generate(state, config, writer=None)

    # greeting path: relevant_docs 가 없고 intent==general 이므로 안내 메시지가 아님.
    # is_cached=False 이므로 cached 단축 경로를 건너뛰고 astream 으로 생성.
    assert len(llm_astream_calls) >= 1
    assert result["response"]  # non-empty answer produced
