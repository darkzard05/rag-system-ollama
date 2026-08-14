"""체크포인터 thread 정리·chat_history 채널 제거·pickle 강등 차단 검증.

TDD 대상 (플랜 투두 7 — R1a-02/R1b-02, R1a-04, R1a-05):
1. 세션 삭제(delete_session / clear_session) 후 그래프 체크포인터에 해당 thread가
   0건 남아 있어야 한다 (InMemorySaver 메모리 누수 방지).
2. GraphState에 chat_history 채널이 없어야 한다 (operator.add 무한 누적 제거).
3. 체크포인터가 pickle_fallback=False를 사용하고, 직렬화 불가 객체 저장 시
   조용한 pickle 강등 대신 명시적 예외가 발생해야 한다.
"""

from typing import Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from api.schemas import GraphState
from core.graph_builder import (
    _graph_cache,
    build_graph,
    delete_graph_thread,
    invalidate_graph_cache,
)
from core.rag_core import RAGSystem
from core.session import SessionManager


class _MiniState(TypedDict):
    """thread 생성용 최소 그래프 상태."""

    input: str


def _echo(state: _MiniState, config: RunnableConfig) -> dict[str, str]:
    """mini 그래프 echo 노드 — 입력에 '!'를 붙여 반환합니다."""
    return {"input": state["input"] + "!"}


def _make_mini_app():
    """체크포인터가 연결된 최소 그래프를 컴파일해 (app, saver)를 반환합니다."""
    saver = InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=False))
    graph = StateGraph(_MiniState)
    graph.add_node("echo", _echo)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=saver), saver


def _thread_checkpoint_count(saver: InMemorySaver, thread_id: str) -> int:
    """해당 thread의 잔존 체크포인트 수를 반환합니다."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return len(list(saver.list(config)))


# ---------------------------------------------------------------- R1a-02/R1b-02


@pytest.mark.asyncio
async def test_delete_session_removes_checkpoint_thread():
    """delete_session 호출 후 체크포인터에 해당 thread가 0건 남아 있어야 한다."""
    sid = "session-td-delete"
    app, saver = _make_mini_app()
    invalidate_graph_cache()
    _graph_cache.checkpointer = saver
    try:
        SessionManager.reset()
        SessionManager.init_session(sid)
        await app.ainvoke({"input": "hi"}, {"configurable": {"thread_id": sid}})
        assert _thread_checkpoint_count(saver, sid) >= 1

        assert SessionManager.delete_session(sid) is True

        assert _thread_checkpoint_count(saver, sid) == 0
    finally:
        invalidate_graph_cache()


@pytest.mark.asyncio
async def test_clear_session_resets_checkpoint_thread():
    """clear_session(reset_all_state 경유) 후 체크포인터 thread가 0건이어야 한다."""
    sid = "session-td-clear"
    app, saver = _make_mini_app()
    invalidate_graph_cache()
    _graph_cache.checkpointer = saver
    try:
        SessionManager.reset()
        await app.ainvoke({"input": "hi"}, {"configurable": {"thread_id": sid}})
        assert _thread_checkpoint_count(saver, sid) >= 1

        RAGSystem(session_id=sid).clear_session()

        assert _thread_checkpoint_count(saver, sid) == 0
    finally:
        invalidate_graph_cache()


def test_delete_graph_thread_missing_thread_is_noop():
    """존재하지 않는 thread 삭제는 예외 없이 통과해야 한다."""
    invalidate_graph_cache()
    try:
        saver = InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=False))
        _graph_cache.checkpointer = saver
        delete_graph_thread("never-existed")
    finally:
        invalidate_graph_cache()


# ------------------------------------------------------------------ R1a-04


def test_graph_state_has_no_chat_history_channel():
    """GraphState에 chat_history 채널이 없어야 한다 (operator.add 누적 제거)."""
    assert "chat_history" not in GraphState.__annotations__


# ------------------------------------------------------------------ R1a-05


@pytest.mark.asyncio
async def test_graph_checkpointer_disables_pickle_fallback():
    """그래프 체크포인터가 pickle_fallback=False를 사용해야 한다."""
    invalidate_graph_cache()
    try:
        with patch("aiosqlite.connect", side_effect=Exception("force InMemorySaver")):
            graph = await build_graph()
            assert graph.checkpointer is not None
            assert graph.checkpointer.serde.pickle_fallback is False
    finally:
        invalidate_graph_cache()


def test_pickle_fallback_false_raises_explicit_error():
    """pickle_fallback=False에서 직렬화 불가 객체는 명시적 예외를 던져야 한다."""
    from core.graph_builder import _sanitize_channel_value

    class _Exotic:
        pass

    serializer = JsonPlusSerializer(pickle_fallback=False)
    with pytest.raises(TypeError):
        serializer.dumps_typed({"bad": _Exotic()})

    with pytest.raises(ValueError, match="pickle 강등 금지"):
        _sanitize_channel_value({"meta": {"o": _Exotic()}})


def test_channel_sanitizer_keeps_pure_types():
    """위생화는 순수 타입(int/str/float/bool/None/list/dict)을 그대로 유지한다."""
    from core.graph_builder import _sanitize_channel_value

    value: Any = {"a": 1, "b": ["x", None, True, 1.5], "c": {"d": "e"}}
    assert _sanitize_channel_value(value) == value


# ------------------------------------------------- R8/R13 — 통합 캐시 폴드 검증


@pytest.mark.asyncio
async def test_invalidate_rebuilds_graph_through_unified_cache():
    """invalidate_graph_cache() 후 build_graph()는 통합 캐시에서 재컴파일해야 한다.

    그래프 항목이 통합 ObjectCache(_graph_object_cache)에 보관되므로,
    invalidate()로 해당 항목이 삭제된 뒤 build_graph()는 새 그래프를 빌드하고
    다시 통합 캐시에 기록해야 한다. 단일 전역 asyncio.Lock + 이중 확인 불변식은
    그대로 preserved 된다.
    """
    from core.graph_builder import _GRAPH_CACHE_KEY, _graph_object_cache

    invalidate_graph_cache()
    await build_graph()

    # 항목이 통합 캐시에 단일 키로 보관되어 있다.
    entry_after_build = await _graph_object_cache.get(_GRAPH_CACHE_KEY)
    assert entry_after_build is not None
    assert entry_after_build.compiled is not None
    assert entry_after_build.checkpointer is not None

    first_compiled = entry_after_build.compiled

    # 무효화 → 항목이 통합 캐시에서 제거된다.
    invalidate_graph_cache()
    assert await _graph_object_cache.get(_GRAPH_CACHE_KEY) is None

    # 재빌드 → 동일한 단일 키로 새 항목이 다시 보관된다.
    recompiled = await build_graph()
    entry_after_rebuild = await _graph_object_cache.get(_GRAPH_CACHE_KEY)
    assert entry_after_rebuild is not None
    # 컴파일 결과는 매 빌드마다 새 객체이므로 identity는 다르다.
    assert entry_after_rebuild.compiled is not first_compiled
    assert recompiled is entry_after_rebuild.compiled

    invalidate_graph_cache()
