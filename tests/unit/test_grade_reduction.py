"""
단위 테스트: grade 단계 축소 (T8 메모이즈 재사용 / T9 opt-out) 검증.

대상:
- core.graph_builder.grade_documents()
  - (T9) GRADING_ENABLED=False → LLM 호출 없이 즉시 {"intent":"generate","route":"generate"} 반환
  - (T5) is_cached=True 단축 경로 → LLM 호출 없이 {"route":"generate"} 반환
  - (T8) 동일 (query, doc_id-set) 반복 호출 시 메모이즈된 판단 재사용 (LLM 1회만 호출)
  - (T8) doc-set 이 바뀌면 메모 miss → LLM 재호출

모든 테스트는 격리된 mock seam 을 사용하며 실제 Ollama/네트워크 호출은 없다.
Mock 대상: ModelManager.inference_session, SessionManager.get/set/add_status_log,
get_active_rerank_engine, adispatch_custom_event, 그리고 config["configurable"]["llm"].
"""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.graph_builder import grade_documents


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _base_state(query: str = "what is RAG?", **overrides: object) -> dict[str, Any]:
    """GraphState 필드에 맞는 상태 dict 생성."""
    state: dict[str, Any] = {
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


def _run_config(session_id: str = "grade-test-session", llm: Any = None) -> dict:
    cfg: dict[str, Any] = {"configurable": {"session_id": session_id}}
    if llm is not None:
        cfg["configurable"]["llm"] = llm
    return cfg


def _docs(doc_ids: list[str]) -> list[Document]:
    """doc_id 가 명시된 Document 셋 생성 (메모 키 안정성 보장)."""
    return [
        Document(page_content=f"chunk content {i}", metadata={"doc_id": d})
        for i, d in enumerate(doc_ids)
    ]


class _FakeSessionStore:
    """SessionManager 의 get/set/add_status_log 을 dict 로 백업하는 페이크.

    grade 단계는 메모(_GRADE_MEMO_KEY)를 세션에 저장/조회하므로, 동일 세션 내
    두 번째 호출에서 이전 판단을 꺼내올 수 있어야 한다.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(
        self,
        key: str,
        default: Any = None,
        session_id: str | None = None,
        create: bool = True,
    ) -> Any:
        return self._data.get(key, default)

    def set(
        self,
        key: str | None = None,
        value: Any = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if key is not None:
            self._data[key] = value
        else:
            self._data.update(kwargs)

    def add_status_log(self, message: str, session_id: str | None = None) -> None:
        return None

    @classmethod
    def get_session_id(cls) -> str:
        return "default"


@asynccontextmanager
async def _patch_grade_deps(decision: dict):
    """grade_documents 실행에 필요한 모든 외부 의존성을 mock 으로 교체.

    yields: (store, llm, json_llm) — json_llm.ainvoke.call_count 로 LLM 호출 횟수를
    단언할 수 있다.
    """
    store = _FakeSessionStore()

    json_llm = AsyncMock()
    json_llm.ainvoke.return_value = SimpleNamespace(content=json.dumps(decision))

    llm = MagicMock()
    llm.bind.return_value = json_llm

    fake_mm = MagicMock()

    @asynccontextmanager
    async def _null_session():
        yield

    fake_mm.inference_session = _null_session  # type: ignore[attr-defined]

    with (
        patch("core.graph_builder.SessionManager", store),
        patch("core.graph_builder.ModelManager", fake_mm),
        patch(
            "core.async_reranker.get_active_rerank_engine",
            return_value="flashrank",
        ),
        patch("core.graph_builder.adispatch_custom_event", new=AsyncMock()),
    ):
        yield store, llm, json_llm


# ----------------------------------------------------------------------------
# (a) memo reuse — same (query, doc-set) → LLM called only ONCE
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_memo_reuse_same_docset_calls_llm_once(monkeypatch):
    """T8: 동일 상태 재호출 시 LLM 호출 횟수 == 1 (메모 히트)."""
    monkeypatch.setattr("core.graph_builder.GRADING_ENABLED", True)

    # rewrite 판단이 메모에 "transform" 으로 저장되도록 설정.
    decision = {
        "action": "rewrite",
        "is_relevant": False,
        "relevant_entities": ["x"],
        "reason": "not relevant",
        "optimized_query": "better query",
    }
    docs = _docs(["doc-a1", "doc-a2"])
    state = _base_state("same query", relevant_docs=docs)

    async with _patch_grade_deps(decision) as (store, llm, json_llm):
        config = _run_config(llm=llm)

        result1 = await grade_documents(state, config, writer=None)
        result2 = await grade_documents(state, config, writer=None)

    # LLM 은 첫 호출에서만 불렸고, 두 번째는 메모에서 재사용.
    assert json_llm.ainvoke.call_count == 1
    # 첫 호출은 실제 grade 결과(route=transform) 를 반환.
    assert result1["route"] == "transform"
    assert result1["intent"] == "transform"
    # 두 번째 호출은 메모이즈된 판단을 재사용 — route 는 transform 과 일치하나,
    # 메모 단축 경로는 intent="generate" 로 정규화한다 (graph_builder:805).
    assert result2["route"] == "transform"
    assert result2["intent"] == "generate"


# ----------------------------------------------------------------------------
# (b) different doc-set → fresh grade (memo miss)
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_different_docset_forces_fresh_grade(monkeypatch):
    """T8: doc_id-set 이 바뀌면 메모 miss → LLM 이 다시 호출된다."""
    monkeypatch.setattr("core.graph_builder.GRADING_ENABLED", True)

    decision = {
        "action": "generate",
        "is_relevant": True,
        "relevant_entities": ["x"],
        "reason": "relevant",
        "optimized_query": None,
    }
    docs_a = _docs(["doc-a1", "doc-a2"])
    docs_b = _docs(["doc-b1", "doc-b2"])
    state_a = _base_state("shared query", relevant_docs=docs_a)
    state_b = _base_state("shared query", relevant_docs=docs_b)

    async with _patch_grade_deps(decision) as (store, llm, json_llm):
        config = _run_config(llm=llm)

        await grade_documents(state_a, config, writer=None)
        await grade_documents(state_b, config, writer=None)

    # doc-set 이 다르므로 두 번 모두 LLM 호출 (메모 미스).
    assert json_llm.ainvoke.call_count == 2


# ----------------------------------------------------------------------------
# (c) opt-out — GRADING_ENABLED=False → route "generate", LLM never called
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_opt_out_grading_disabled_returns_generate_no_llm(monkeypatch):
    """T9: GRADING_ENABLED=False → LLM 미호출, {"intent":"generate","route":"generate"}."""
    monkeypatch.setattr("core.graph_builder.GRADING_ENABLED", False)

    docs = _docs(["doc-a1", "doc-a2"])
    state = _base_state("any query", relevant_docs=docs)

    async with _patch_grade_deps(
        {
            "action": "generate",
            "is_relevant": True,
            "relevant_entities": [],
            "reason": "x",
            "optimized_query": None,
        }
    ) as (store, llm, json_llm):
        config = _run_config(llm=llm)
        result = await grade_documents(state, config, writer=None)

    assert result["route"] == "generate"
    assert result["intent"] == "generate"
    # opt-out 은 LLM 경로보다 우선 — 호출 0회.
    assert json_llm.ainvoke.call_count == 0


# ----------------------------------------------------------------------------
# (d) guard preservation
#   (d1) is_cached short-circuit (T5) intact: no LLM, route "generate"
#   (d2) normal path still reaches LLM and returns a valid route
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_is_cached_short_circuit_skips_llm(monkeypatch):
    """T5: is_cached=True 단축 경로는 LLM 을 호출하지 않고 {"route":"generate"} 반환."""
    monkeypatch.setattr("core.graph_builder.GRADING_ENABLED", True)

    docs = _docs(["doc-a1", "doc-a2"])
    state = _base_state("any query", relevant_docs=docs, is_cached=True)

    async with _patch_grade_deps(
        {
            "action": "generate",
            "is_relevant": True,
            "relevant_entities": [],
            "reason": "x",
            "optimized_query": None,
        }
    ) as (store, llm, json_llm):
        config = _run_config(llm=llm)
        result = await grade_documents(state, config, writer=None)

    assert result["route"] == "generate"
    assert json_llm.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_normal_path_reaches_llm_and_returns_valid_route(monkeypatch):
    """안전 단축 없이 일반 경로가 LLM 에 도달하고 유효한 route 를 반환."""
    monkeypatch.setattr("core.graph_builder.GRADING_ENABLED", True)

    decision = {
        "action": "generate",
        "is_relevant": True,
        "relevant_entities": ["x"],
        "reason": "relevant enough",
        "optimized_query": None,
    }
    docs = _docs(["doc-a1", "doc-a2"])
    state = _base_state("a normal query", relevant_docs=docs)

    async with _patch_grade_deps(decision) as (store, llm, json_llm):
        config = _run_config(llm=llm)
        result = await grade_documents(state, config, writer=None)

    # 일반 경로: LLM 이 실제로 호출되었고 유효한 라우트가 반환됨.
    assert json_llm.ainvoke.call_count >= 1
    assert result["route"] in ("generate", "transform")
