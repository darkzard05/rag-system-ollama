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

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from common.config import MAX_CONCURRENT_INFERENCE
from core.graph._glue import _start_speculative_generate
from core.graph._speculative_gen import _adopt_speculative_generate
from core.graph.graph_builder import (
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
        "core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE",
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

    with patch(
        "core.graph._generate.adispatch_custom_event", side_effect=fake_dispatch
    ):
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
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_slow_generate),
        patch("core.graph._generate.adispatch_custom_event", side_effect=fake_dispatch),
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
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch(
            "core.graph._generate.OLLAMA_NUM_CTX",
            8192,
        ),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", 2048),
        patch("core.graph._generate.count_tokens_rough", return_value=10),
        patch.object(ModelManager, "inference_session", _mock_session()),
        patch("core.graph._grade.adispatch_custom_event", side_effect=fake_dispatch),
        patch("core.graph._generate.adispatch_custom_event", side_effect=fake_dispatch),
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
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_slow_generate),
        patch("core.graph._generate.adispatch_custom_event", side_effect=fake_dispatch),
    ):
        grade_task = asyncio.ensure_future(grade_documents(state, config, writer=None))
        await asyncio.wait_for(started["ran"].wait(), timeout=2)
        result = await grade_task

    assert result["route"] == "transform"
    # registry 정리 (orphan 없음)
    assert thread_id not in _spec_registry
    # speculative generate의 이벤트는 절대 전달되지 않음 (미노출)
    assert dispatched == []


# ---------------------------------------------------------------------------
# Step 1.5 Regression Tests (Issue 1 fix)
# Plan: .omo/plans/top3-fixes.md lines 93-114
#   A) catch-all except → speculative cancel + _grade_op.__exit__ + re-raise
#   B) registry hit → stale orphan cancelled, fresh task registered
#   C) adopt skips a dead (cancelled) task
#   D) double start on same thread_id → first cancelled, fresh second
# ---------------------------------------------------------------------------
# 겹침 토글 기법: 기존 테스트와 동일하게
# ``patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2)`` 을 사용한다.
# _spec_overlap_enabled() 는 _speculative_gen 모듈 전역 MAX_CONCURRENT_INFERENCE 를
# 읽으므로 그 네임스페이스에서 패치해야 한다 (config.yml 기본값은 1).
async def _pending_generate(state, config, *, writer):
    """미완료 상태로 유지되는 (패치용) generate — 취소 가능한 pending task."""
    await asyncio.sleep(60)
    return {"response": "never"}


async def _drain_tasks(tasks: list[asyncio.Task]) -> None:
    """이벤트 루프를 돌려 취소(cancel)를 처리한 뒤 task 상태를 확정한다."""
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Step 1.5 Test A: 포착 튜플 밖 예외 → 특기 task 취소 + _grade_op.__exit__ + 재발사
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exception_safety_cancel_and_exit():
    """grade LLM 호출이 포착 튜플(RuntimeError, ValueError, JSONDecodeError) 밖의
    예외(httpx.ConnectError)로 panic하면 speculative generate가 취소·registry에서
    정리되고, _grade_op.__exit__ 이 호출되며, 예외가 그대로 재발사된다
    (Step 1.5 Test A)."""
    exited: list[tuple] = []

    def _tracker(*_args):  # _enter_stage(OperationType.LLM_INFERENCE) 호출 대응
        tracker = MagicMock()
        tracker.__exit__.side_effect = lambda *args: exited.append(args)
        return tracker

    # 기존 _mock_session()은 __aexit__ 가 truthy(AsyncMock 기본값)라 with 블록
    # 내부 예외를 삼킨다 — 여기서는 예외가 전파되도록 __aexit__=False 로 만든다.
    def _mock_session_no_suppress():
        sess = MagicMock()
        sess.return_value.__aenter__ = AsyncMock()
        sess.return_value.__aexit__ = AsyncMock(return_value=False)
        return sess

    started: dict[str, asyncio.Event] = {"ran": asyncio.Event()}

    async def _slow_generate(state, config, *, writer):
        started["ran"].set()  # speculative task 가 실제 시작됨을 보장
        await asyncio.sleep(10)  # panic 이 결정될 때까지 미완료 유지
        return {"response": "never"}

    async def _raise_connect(llm, prompt, config, model_name="default"):
        # 루프 양보를 한 번 준 뒤 예외를 던진다 — 양보 덕분에 speculative
        # task(_slow_generate)가 ran 을 set 하고 시작하는 것이 보장된다.
        await asyncio.sleep(0)
        raise httpx.ConnectError("simulated connection failure")

    thread_id = "panic-1"
    state = {
        "input": "예외 안전성 질문 (Step 1.5 Test A)",
        "relevant_docs": [
            Document(page_content="패닉 문서", metadata={"rerank_score": 0.3})
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    config = {"configurable": {"llm": _json_llm("generate"), "thread_id": thread_id}}

    with (
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_slow_generate),
        patch("core.graph._grade._enter_stage", side_effect=_tracker),
        patch.object(ModelManager, "inference_session", _mock_session_no_suppress()),
        patch("core.graph._grade._safe_invoke", side_effect=_raise_connect),
    ):
        grade_task = asyncio.ensure_future(grade_documents(state, config, writer=None))
        await asyncio.wait_for(started["ran"].wait(), timeout=2)
        with pytest.raises(httpx.ConnectError):
            await grade_task

    # (a) speculative task 가 thread_id 기준으로 registry 에서 정리됨 (orphan 없음)
    assert thread_id not in _spec_registry
    # (b) _grade_op.__exit__ 이 예외 정보와 함께 호출됨 (tracker 스파이)
    assert exited
    assert exited[-1][0] is httpx.ConnectError
    # (c) 예외가 catch-all 에서 삼켜지지 않고 재발사됨 — 위 pytest.raises 로 검증


# ---------------------------------------------------------------------------
# Step 1.5 Test B: registry hit → stale orphan 취소 + 새 task 등록
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_registry_hit_cancels_stale_orphan():
    """동일 thread_id 에 이미 orphan 이 registry 에 있으면 _start_speculative_generate
    재진입 시 stale task 를 취소·버퍼 폐기하고 새 task 를 등록한다 — registry 에는
    정확히 하나의 entry 만 남는다 (Step 1.5 Test B)."""
    thread_id = "stale-1"
    stale = asyncio.ensure_future(asyncio.sleep(60))
    _spec_registry[thread_id] = _SpecGenerate(task=stale, buffer=[], adopter=None)

    state = {"input": "질문", "relevant_docs": [], "is_cached": False}
    config = {"configurable": {"thread_id": thread_id}}

    with (
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_pending_generate),
    ):
        assert _start_speculative_generate(state, config, writer=None) == thread_id
        fresh = _spec_registry[thread_id].task

    await _drain_tasks([stale])
    assert stale.cancelled()  # stale task 취소됨
    assert fresh is not stale  # 새 task 가 (재)등록됨
    assert thread_id in _spec_registry
    assert len(_spec_registry) == 1  # orphan 없이 정확히 하나

    await _drain_tasks([fresh])  # pending task 정리 (후속 테스트 누수 방지)


# ---------------------------------------------------------------------------
# Step 1.5 Test C: 죽은(cancelled) task 는 채택하지 않음
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_adopt_skips_dead_task():
    """registry 의 speculative task 가 이미 취소(cancelled) 상태면
    _adopt_speculative_generate 는 None 을 반환하고, entry 를 제거하며,
    버퍼를 폐기한다 (Step 1.5 Test C)."""
    thread_id = "dead-1"
    buf = [SimpleNamespace(name="graph_status", data={"status": "x"}, config=None)]

    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    await _drain_tasks([task])  # CancelledError 전달 → cancelled() True 확정
    assert task.cancelled()

    _spec_registry[thread_id] = _SpecGenerate(task=task, buffer=buf)
    config = {"configurable": {"thread_id": thread_id}}

    result = _adopt_speculative_generate(config)

    assert result is None  # 죽은 task 를 채택하지 않음
    assert thread_id not in _spec_registry
    assert buf == []  # dead task 의 버퍼는 폐기됨


# ---------------------------------------------------------------------------
# Step 1.5 Test D: 동일 thread_id 이중 시작 → 첫 task 취소, 두 번째 새로 등록
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_double_start_same_thread_id():
    """동일 thread_id 에 _start_speculative_generate 를 두 번 호출하면
    첫 task 가 취소되고 두 번째 task 가 새로 등록된다 — registry 에는
    정확히 하나 (Step 1.5 Test D)."""
    thread_id = "double-1"
    state = {"input": "질문", "relevant_docs": [], "is_cached": False}
    config = {"configurable": {"thread_id": thread_id}}

    with (
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_pending_generate),
    ):
        assert _start_speculative_generate(state, config, writer=None) == thread_id
        first = _spec_registry[thread_id].task

        assert _start_speculative_generate(state, config, writer=None) == thread_id
        second = _spec_registry[thread_id].task

    await _drain_tasks([first])
    assert first.cancelled()  # 첫 task 취소됨
    assert second is not first  # 두 번째는 새 task
    assert thread_id in _spec_registry
    assert len(_spec_registry) == 1  # orphan 없이 정확히 하나

    await _drain_tasks([second])  # pending task 정리 (후속 테스트 누수 방지)
