"""
단위 테스트: 쿼리 캐시 키 네임스페이싱 (D17).

D17 은 쿼리 응답 캐시 키를 ``f"{file_hash}:{query}"`` 로 네임스페이싱하여
같은 질의 문자열이 다른 문서에서 조회/저장될 때 서로 충돌하지 않게 한다.
(교차 문서 의미 충돌 → 이전 문서의 캐시 답변 참조 버그 방지)

검증 대상:
- core.graph._preprocess.preprocess()  : get 경로가 네임스페이스 키를 사용
- core.graph._generate.generate()      : set 경로가 네임스페이스 키를 사용
- 서로 다른 file_hash 가 같은 질의에 대해 서로 다른 키를 생성 (비충돌)

기존 tests/unit/test_query_cache.py 의 패턴(모의 seam, fixture, 하네스)을
그대로 따른다 — Ollama/네트워크 호출은 없다.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph.graph_builder import generate, preprocess
from core.session import SessionManager


# ----------------------------------------------------------------------------
# Fixtures / helpers (test_query_cache.py 의 패턴을 그대로 재사용)
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


def _make_cache_manager(get_return: object) -> MagicMock:
    """get() 이 get_return 을 반환하는 CacheManager 페이크."""
    cm = MagicMock()
    cm.get = AsyncMock(return_value=get_return)
    cm.set = AsyncMock()
    return cm


# ----------------------------------------------------------------------------
# (a) get 경로: file_hash 가 네임스페이스 키로 조회된다
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_uses_namespaced_key(monkeypatch):
    """file_hash="abc123" 세션에서 preprocess 의 get() 은 "abc123:<query>" 로 호출."""
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_MIN_CONF", 0.85)

    sid = "ns-session-get"
    SessionManager.set("file_hash", "abc123", session_id=sid)

    fake_cm = _make_cache_manager({"response": "cached answer", "confidence": 0.99})
    state = _base_state("reusable query")
    try:
        with (
            patch("core.graph._preprocess.get_cache_manager", return_value=fake_cm),
            patch(
                "core.graph._preprocess._ensure_query_cache_embedder",
                new=AsyncMock(),
            ),
        ):
            result = await preprocess(state, _base_config(sid), writer=None)

        fake_cm.get.assert_called_once_with("abc123:reusable query", use_semantic=True)
        assert result["is_cached"] is True
    finally:
        SessionManager.delete("file_hash", session_id=sid)


# ----------------------------------------------------------------------------
# (b) set 경로: file_hash 가 네임스페이스 키로 저장된다
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_set_uses_namespaced_key(monkeypatch):
    """file_hash="abc123" 세션에서 generate 의 set() 은 "abc123:<query>" 로 호출."""
    monkeypatch.setattr("core.graph._generate.QUERY_CACHE_ENABLED", True)
    # 저장 게이트(신뢰도)만 낮춰 폴백 confidence(0.5)가 통과하도록 한다.
    monkeypatch.setattr("core.graph._generate.QUERY_CACHE_MIN_CONF", 0.0)

    sid = "ns-session-set"
    SessionManager.set("file_hash", "abc123", session_id=sid)

    llm = MagicMock()
    llm.bind.return_value = llm

    async def fake_astream(*args, **kwargs):
        chunk = SimpleNamespace(content="hello there", response_metadata={})
        yield chunk

    llm.astream = fake_astream
    llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, None)

    state = _base_state(
        "reusable query",
        intent="general",
        is_cached=False,
        cached_response=None,
        relevant_docs=[],
    )
    config = _base_config(sid)
    config["configurable"]["llm"] = llm

    fake_cm = _make_cache_manager(None)
    captured_events: list[dict] = []
    try:
        with (
            patch("core.graph._generate.get_cache_manager", return_value=fake_cm),
            patch(
                "core.graph._generate._ensure_query_cache_embedder",
                new=AsyncMock(),
            ),
            patch(
                "core.graph._glue.adispatch_custom_event",
                side_effect=lambda name, data, config=None: captured_events.append(
                    {"name": name, "data": dict(data)}
                ),
            ),
        ):
            result = await generate(state, config, writer=None)

        fake_cm.set.assert_called_once()
        (key,) = fake_cm.set.call_args.args[:1]
        assert key == "abc123:reusable query"
        assert result["response"]
    finally:
        SessionManager.delete("file_hash", session_id=sid)


# ----------------------------------------------------------------------------
# (c) 서로 다른 file_hash 는 같은 질의라도 다른 키를 생성한다
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_different_file_hashes_do_not_collide(monkeypatch):
    """같은 질의 문자열이어도 file_hash 가 다르면 get() 키가 달라야 한다."""
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_MIN_CONF", 0.85)

    keys: list[str] = []
    for sid, file_hash in (("ns-session-a", "abc123"), ("ns-session-b", "def456")):
        SessionManager.set("file_hash", file_hash, session_id=sid)
        fake_cm = _make_cache_manager({"response": "cached answer", "confidence": 0.99})
        state = _base_state("reusable query")
        try:
            with (
                patch(
                    "core.graph._preprocess.get_cache_manager",
                    return_value=fake_cm,
                ),
                patch(
                    "core.graph._preprocess._ensure_query_cache_embedder",
                    new=AsyncMock(),
                ),
            ):
                await preprocess(state, _base_config(sid), writer=None)
            keys.append(fake_cm.get.call_args.args[0])
        finally:
            SessionManager.delete("file_hash", session_id=sid)

    assert keys == ["abc123:reusable query", "def456:reusable query"]
    assert keys[0] != keys[1]
