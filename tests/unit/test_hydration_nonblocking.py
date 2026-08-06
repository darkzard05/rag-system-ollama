"""하이드레이션 비차단화 검증 테스트 (계획 Todo 8).

astream `_consumer`의 retrieve 청크 처리에서 `hydrate_documents`가 inline
`await`가 아닌 `asyncio.create_task`로 스케줄되어 첫 토큰 전 지연을 유발하지
않고, `_consumer` 종료(finally) 시 모든 태스크가 try/except로 await되어
예외가 재발생하지 않는지 검증합니다. rewrite 사이클로 retrieve 청크가 2회
발생하는 케이스도 포함합니다.
"""

import asyncio
import gc
import time
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.rag_core import RAGSystem


def _retrieve_event(docs: list[Document]) -> dict[str, Any]:
    """retrieve 노드의 on_chain_stream 이벤트를 구성합니다."""
    return {
        "event": "on_chain_stream",
        "metadata": {},
        "data": {"chunk": {"retrieve": {"relevant_docs": docs}}},
    }


def _generate_event() -> dict[str, Any]:
    """generate 노드의 on_chain_stream 이벤트 (스트림 종료 전 마지막 이벤트)."""
    return {
        "event": "on_chain_stream",
        "metadata": {},
        "data": {"chunk": {"generate": {"content": "답변"}}},
    }


def _stream(events: list[dict[str, Any]]):
    """주어진 이벤트를 yield하는 async 제너레이터 팩토리를 반환합니다."""

    async def fake_stream(*args: Any, **kwargs: Any):
        for event in events:
            yield event

    return fake_stream


@pytest.fixture
def astream_env() -> Iterator[dict[str, Any]]:
    """astream의 의존성을 모두 모킹한 환경을 구성합니다.

    SessionManager / prepare_query_config_or_build / _get_rag_engine /
    circuit breaker / resource manager를 패치하고, `hydrate_documents`는
    테스트별로 side_effect 제어를 위해 별도 패치합니다.
    """
    rag = RAGSystem(session_id="test_hydration_nonblocking")
    engine = AsyncMock()
    breaker = MagicMock()
    registry = MagicMock()
    registry.get_breaker.return_value = breaker
    manager = MagicMock()
    session_manager = MagicMock()
    session_manager.get.return_value = None
    session_manager.get_messages.return_value = []

    env: dict[str, Any] = {"rag": rag, "breaker": breaker, "manager": manager}

    with ExitStack() as stack:
        stack.enter_context(patch("core.rag_core.SessionManager", session_manager))
        stack.enter_context(
            patch(
                "core.rag_core.prepare_query_config_or_build",
                AsyncMock(return_value={}),
            )
        )
        stack.enter_context(
            patch.object(rag, "_get_rag_engine", AsyncMock(return_value=engine))
        )
        stack.enter_context(
            patch("core.rag_core.get_circuit_breaker_registry", return_value=registry)
        )
        stack.enter_context(
            patch("core.rag_core.get_resource_manager", return_value=manager)
        )
        yield env


async def _drain(gen: Any) -> None:
    """async 제너레이터를 끝까지 소비해 finally 실행까지 마칩니다."""
    while True:
        try:
            await anext(gen)
        except StopAsyncIteration:
            return


@pytest.mark.asyncio
async def test_hydration_scheduled_as_task_nonblocking(astream_env):
    """retrieve 청크에서 hydrate_documents가 create_task로 스케줄되어 첫 토큰을 지연시키지 않아야 합니다."""
    rag = astream_env["rag"]
    breaker = astream_env["breaker"]
    docs = [
        Document(
            page_content="본문",
            metadata={"file_path": "doc.pdf", "has_coordinates": True, "page": 1},
        )
    ]
    started = asyncio.Event()
    finished = asyncio.Event()

    async def slow_hydrate(docs_list):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    breaker.call_async_stream = lambda *args, **kwargs: _stream(
        [_retrieve_event(docs), _generate_event()]
    )()

    with patch(
        "core.rag_core.hydrate_documents", side_effect=slow_hydrate
    ) as mock_hydrate:
        gen = await rag.astream("질문")
        t0 = time.monotonic()
        kind, _chunk = await anext(gen)
        elapsed = time.monotonic() - t0

        assert kind == "updates"
        assert elapsed < 0.02, (
            f"hydrate_documents가 inline await로 첫 토큰을 지연시켰습니다 "
            f"(elapsed={elapsed:.3f}s, hydration sleep=0.05s)"
        )
        # 소비자가 첫 이벤트를 yield한 뒤 루프가 스케줄된 태스크를 실행 → 병렬 실행 증명
        await asyncio.sleep(0)
        assert started.is_set(), (
            "hydrate_documents가 create_task로 스케줄되지 않았습니다"
        )
        mock_hydrate.assert_called_once()

        await _drain(gen)

    assert finished.is_set(), "finally가 hydration 태스크를 await하지 않았습니다"
    astream_env["manager"].unpin_retrievers.assert_called_once()


@pytest.mark.asyncio
async def test_hydration_failure_swallowed_by_finally(astream_env, caplog):
    """hydration 실패가 스트림 실패로 승격되지 않고 로그-온리로 처리되어야 합니다."""
    rag = astream_env["rag"]
    breaker = astream_env["breaker"]

    async def failing_hydrate(docs_list):
        raise RuntimeError("hydration boom")

    breaker.call_async_stream = lambda *args, **kwargs: _stream(
        [
            _retrieve_event([Document(page_content="본문", metadata={})]),
            _generate_event(),
        ]
    )()

    with patch("core.rag_core.hydrate_documents", side_effect=failing_hydrate):
        gen = await rag.astream("질문")
        kind, _chunk = await anext(gen)
        assert kind == "updates"

        await _drain(gen)

    # 소비자 스트림이 실패하지 않았고, 실패는 로그로만 남았다.
    assert "문서 하이드레이션 실패" in caplog.text
    # 태스크 예외가 미소비 상태로 남지 않았다 (never retrieved 경고 부재).
    gc.collect()
    assert "never retrieved" not in caplog.text


@pytest.mark.asyncio
async def test_rewrite_cycle_schedules_and_awaits_two_tasks(astream_env):
    """rewrite 루프로 retrieve 청크가 2회 발생해도 태스크 2개가 모두 보관·await되어야 합니다."""
    rag = astream_env["rag"]
    breaker = astream_env["breaker"]
    docs_a = [Document(page_content="문서 A", metadata={})]
    docs_b = [Document(page_content="문서 B", metadata={})]
    completed: list[list[Document]] = []

    async def recording_hydrate(docs_list):
        await asyncio.sleep(0.01)
        completed.append(docs_list)

    breaker.call_async_stream = lambda *args, **kwargs: _stream(
        [_retrieve_event(docs_a), _retrieve_event(docs_b), _generate_event()]
    )()

    with patch(
        "core.rag_core.hydrate_documents", side_effect=recording_hydrate
    ) as mock_hydrate:
        gen = await rag.astream("질문")
        await _drain(gen)

    assert mock_hydrate.call_count == 2
    assert completed == [docs_a, docs_b]
