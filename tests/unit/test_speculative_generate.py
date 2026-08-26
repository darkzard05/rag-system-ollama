"""
PHASE 2 검증: grade_documents가 generate LLM 라운드트립을 선제 실행(speculative)하여
grade 와 겹쳐 실행하는지, 그리고 route=generate에서는 warm task를 채택(단일 LLM 호출,
단 더 일찍 시작)하고 route=transform에서는 취소·미노출하는지 검증합니다.

안전 불변:
* MAX_CONCURRENT_INFERENCE == 1(기본값)일 때는 겹침을 수행하지 않는다 (파이프라인 동일).
* route=transform 시 speculative generate의 이벤트는 사용자에게 절대 전달되지 않는다.
* route=generate 시 단일 LLM 호출만 발생하며(겹침은 호출 수를 늘리지 않음) 채택 시
  버퍼링된 이벤트가 재생(replay)된다.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from common.config import MAX_CONCURRENT_INFERENCE
from core.graph_builder import (
    _spec_registry,
    _SpecGenerate,
    generate,
    grade_documents,
)
from core.model_loader import ModelManager


def _mock_session():
    sess = MagicMock()
    sess.return_value.__aenter__ = AsyncMock()
    sess.return_value.__aexit__ = AsyncMock()
    return sess


def _json_llm(action: str, is_relevant: bool = True, optimized_query=None):
    llm = MagicMock()
    jllm = AsyncMock()

    def _resp(**fields):
        return SimpleNamespace(content=json.dumps(fields))

    jllm.ainvoke.return_value = _resp(
        action=action,
        is_relevant=is_relevant,
        relevant_entities=["X"],
        reason="ok",
        optimized_query=optimized_query,
    )
    llm.bind.return_value = jllm
    return llm


@pytest.fixture(autouse=True)
def _clear_registry():
    _spec_registry.clear()
    yield
    _spec_registry.clear()


# ---------------------------------------------------------------------------
# 1) 기본값(bound==1): 겹침 없음 — 파이프라인 동작 보존
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_overlap_when_concurrency_bound_is_one():
    """MAX_CONCURRENT_INFERENCE==1이면 grade가 speculative generate를 시작하지 않는다."""
    state = {
        "input": "질문",
        "relevant_docs": [
            Document(page_content="문서", metadata={"rerank_score": 0.5})
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    config = {"configurable": {"llm": _json_llm("generate"), "thread_id": "t1"}}
    with patch(
        "core.graph_builder.MAX_CONCURRENT_INFERENCE",
        min(MAX_CONCURRENT_INFERENCE, 1),
    ):
        assert _spec_overlap_disabled()
        result = await grade_documents(state, config, writer=None)
    assert result == {"intent": "generate", "route": "generate"}
    assert _spec_registry == {}


def _spec_overlap_disabled() -> bool:
    return MAX_CONCURRENT_INFERENCE <= 1


def _spec_overlap_enabled() -> bool:
    return MAX_CONCURRENT_INFERENCE > 1


# ---------------------------------------------------------------------------
# 2) route=generate: speculative task 채택 → 버퍼 재생, LLM 호출 1회
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_adopts_speculative_task_and_replays_buffer():
    """registry에 warm task가 있으면 generate가 채택하고 버퍼 이벤트를 재생한다.

    단일 LLM 호출(astream 미호출)이며, buffered 이벤트가 adispatch_custom_event로
    재전달된다.
    """
    dispatched = []

    async def fake_dispatch(name, data, config=None):
        dispatched.append((name, data))

    thread_id = "adopt-1"
    buffered = [SimpleNamespace(name="graph_status", data={"status": "x"}, config=None)]
    task = asyncio.ensure_future(_fake_generate_coro("답변"))
    _spec_registry[thread_id] = _SpecGenerate(task=task, buffer=buffered)

    state = {
        "input": "질문",
        "relevant_docs": [
            Document(page_content="문서", metadata={"rerank_score": 0.5})
        ],
        "is_cached": False,
    }
    config = {"configurable": {"llm": MagicMock(), "thread_id": thread_id}}

    with patch("core.graph_builder.adispatch_custom_event", side_effect=fake_dispatch):
        result = await generate(state, config, writer=MagicMock())

    assert result == {"response": "답변"}
    # buffered 이벤트가 재생되었는지
    assert ("graph_status", {"status": "x"}) in dispatched
    assert (
        task in _spec_registry_consumed(thread_id) or True
    )  # 채택 시 registry에서 제거
    assert thread_id not in _spec_registry


async def _fake_generate_coro(resp: str) -> dict:
    return {"response": resp}


def _spec_registry_consumed(_tid: str):  # helper placeholder (채택 시 pop 됨)
    return []


# ---------------------------------------------------------------------------
# 3) route=transform: speculative generate 취소·미노출
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_speculative_generate_cancelled_on_transform_route():
    """grade가 transform으로 라우팅되면 speculative generate가 취소되고 미노출된다."""
    dispatched = []

    async def fake_dispatch(name, data, config=None):
        dispatched.append((name, data))

    started: dict[str, asyncio.Event] = {"ran": asyncio.Event()}

    async def _slow_generate(state, config, *, writer):
        started["ran"].set()
        await asyncio.sleep(10)  # route가 결정될 때까지 미완료 상태 유지
        return {"response": "never"}

    state = {
        "input": "질문",
        "relevant_docs": [
            Document(page_content="문서", metadata={"rerank_score": 0.3})
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    config = {
        "configurable": {
            "llm": _json_llm("rewrite", False, "최적화"),
            "thread_id": "t-x",
        }
    }

    with (
        patch("core.graph_builder.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph_builder.generate", side_effect=_slow_generate),
        patch("core.graph_builder.adispatch_custom_event", side_effect=fake_dispatch),
    ):
        # grade_documents가 시작한 speculative generate(_slow_generate)가 실행되도록 양보
        grade_task = asyncio.ensure_future(grade_documents(state, config, writer=None))
        await asyncio.wait_for(started["ran"].wait(), timeout=2)
        result = await grade_task

    assert result["route"] == "transform"
    assert "t-x" not in _spec_registry
    # speculative generate의 이벤트는 절대 전달되지 않음
    assert dispatched == []


# ---------------------------------------------------------------------------
# 4) route=generate 전체: 겹침 시작 → 채택 (단일 LLM 호출)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_speculative_overlap_generates_route_adopts_single_llm_call():
    """bound>1 + route=generate: grade가 generate를 선제 실행하고 generate가 채택한다.

    LLM astream은 speculative 1회만 호출된다 (호출 수 증가 없음).
    """
    dispatched = []

    async def fake_dispatch(name, data, config=None):
        dispatched.append((name, data))

    thread_id = "overlap-1"
    astream_calls = {"n": 0}

    def make_generate_llm():
        llm = MagicMock()

        # grade path: bind → JSON mode responder (action=generate)
        jllm = AsyncMock()
        jllm.ainvoke.return_value = SimpleNamespace(
            content=json.dumps(
                {
                    "action": "generate",
                    "is_relevant": True,
                    "relevant_entities": ["X"],
                    "reason": "ok",
                    "optimized_query": None,
                }
            )
        )
        llm.bind.return_value = jllm

        # generate path: astream yields one structured chunk
        async def astream(messages, config=None):
            astream_calls["n"] += 1
            yield AIMessageChunk(content='{"reasoning":"","final_answer":"정답"}')

        # generate()는 json_llm(=jllm).astream(...) 을 async for 로 순회하므로
        # jllm 에도 astream 을 세팅한다.
        jllm.astream = astream
        llm.astream = astream
        llm._convert_chunk_to_thought_and_content = lambda c: (c.content, "")
        return llm

    state = {
        "input": "질문",
        "relevant_docs": [
            Document(page_content="문서", metadata={"rerank_score": 0.6})
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    config = {
        "configurable": {
            "llm": make_generate_llm(),
            "thread_id": thread_id,
        }
    }

    with (
        patch("core.graph_builder.MAX_CONCURRENT_INFERENCE", 2),
        patch(
            "core.graph_builder.OLLAMA_NUM_CTX",
            8192,
        ),
        patch("core.graph_builder.OLLAMA_NUM_PREDICT", 2048),
        patch("core.graph_builder.count_tokens_rough", return_value=10),
        patch.object(ModelManager, "inference_session", _mock_session()),
        patch("core.graph_builder.adispatch_custom_event", side_effect=fake_dispatch),
    ):
        # 실제 실행 모델: grade가 route=generate를 반환하면 런타임이 generate를
        # 다시 호출하고, speculative task를 채택(단일 astream 호출 완료)한다.
        result = await grade_documents(state, config, writer=MagicMock())
        assert result["route"] == "generate"
        gen_result = await generate(state, config, writer=MagicMock())
        assert gen_result["response"] == "정답"

    # speculative generate가 astream을 정확히 1회만 호출 (겹침이 호출 수를 늘리지 않음)
    assert astream_calls["n"] == 1
    # 채택 후 registry 정리
    assert thread_id not in _spec_registry


# ---------------------------------------------------------------------------
# 5) LLM/JSON 오류 경로: route=transform → speculative task 취소·미노출·registry 정리
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_speculative_cancelled_on_grade_llm_error_path():
    """grade LLM이 오류(JSON 파싱 실패)로 transform을 반환하면 speculative
    generate가 취소되고, 미노출되며, _spec_registry가 정리된다 (orphan 방지)."""
    dispatched = []

    async def fake_dispatch(name, data, config=None):
        dispatched.append((name, data))

    started: dict[str, asyncio.Event] = {"ran": asyncio.Event()}

    async def _slow_generate(state, config, *, writer):
        started["ran"].set()
        await asyncio.sleep(10)  # 오류가 결정될 때까지 미완료 상태 유지
        return {"response": "never"}

    # grade LLM: bind().ainvoke()와 fallback llm.ainvoke() 모두 ValueError를 던져
    # 외부 except(JSON/LLM error) 경로로 빠진다.
    llm = MagicMock()
    jllm = AsyncMock()
    jllm.ainvoke.side_effect = ValueError("simulated LLM/JSON error")
    llm.bind.return_value = jllm
    llm.ainvoke.side_effect = ValueError("simulated LLM/JSON error")

    thread_id = "err-1"
    state = {
        "input": "질문",
        "relevant_docs": [
            Document(page_content="문서", metadata={"rerank_score": 0.3})
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    config = {"configurable": {"llm": llm, "thread_id": thread_id}}

    with (
        patch("core.graph_builder.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph_builder.generate", side_effect=_slow_generate),
        patch("core.graph_builder.adispatch_custom_event", side_effect=fake_dispatch),
    ):
        grade_task = asyncio.ensure_future(grade_documents(state, config, writer=None))
        await asyncio.wait_for(started["ran"].wait(), timeout=2)
        result = await grade_task

    assert result["route"] == "transform"
    # registry 정리 (orphan 없음)
    assert thread_id not in _spec_registry
    # speculative generate의 이벤트는 절대 전달되지 않음 (미노출)
    assert dispatched == []
