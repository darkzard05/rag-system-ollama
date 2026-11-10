"""QA trace(JSONL) 기록 및 RAG 훅 동작 검증."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

import common.config as cfg
from common import qa_trace
from core.rag_core import RAGSystem


@pytest.fixture
def qa_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "qa_trace.jsonl"
    monkeypatch.setattr(cfg, "QA_LOG_ENABLED", True)
    monkeypatch.setattr(cfg, "QA_LOG_PATH", str(path))
    monkeypatch.setattr(cfg, "QA_LOG_MAX_ANSWER_CHARS", 0)
    monkeypatch.setattr(cfg, "QA_LOG_ROTATE_MAX_BYTES", 0)
    monkeypatch.setattr(cfg, "QA_LOG_ROTATE_BACKUPS", 0)
    qa_trace.reset_for_tests()
    yield path
    qa_trace.reset_for_tests()


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_build_record_maps_fields() -> None:
    record = qa_trace.build_qa_record(
        source="aquery",
        session_id="sid",
        query="질문",
        answer="답변",
        model="model",
        citations=[{"doc_id": "d1"}],
        confidence=0.9,
        prompt_version="v1",
        is_cached=True,
        regenerated="echo",
        parse_failed=True,
        echo_detected=False,
        over_budget=True,
        intent="rag",
        relevant_docs=[
            Document(page_content="x", metadata={"stable_id": "chunk-1"}),
            {"metadata": {"source": "doc.pdf"}},
        ],
        performance={
            "input_token_count": 10,
            "output_token_count": 20,
            "generate_ms": 1.5,
            "ttft_ms": 0.5,
        },
        file_name="doc.pdf",
    )
    assert record["schema_version"] == qa_trace.QA_TRACE_SCHEMA_VERSION
    assert record["source"] == "aquery"
    assert record["is_cached"] is True
    assert record["regenerated"] == "echo"
    assert record["parse_failed"] is True
    assert record["over_budget"] is True
    assert record["relevant_doc_ids"] == ["chunk-1", "doc.pdf"]
    assert record["input_tokens"] == 10
    assert record["output_tokens"] == 20
    assert record["generate_ms"] == 1.5
    assert record["file_name"] == "doc.pdf"


def test_disabled_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "disabled.jsonl"
    monkeypatch.setattr(cfg, "QA_LOG_ENABLED", False)
    monkeypatch.setattr(cfg, "QA_LOG_PATH", str(path))
    qa_trace.reset_for_tests()
    qa_trace.record_qa_turn(source="aquery", session_id="s", query="q", answer="a")
    qa_trace.reset_for_tests()
    assert not path.exists()


def test_writes_single_jsonl_line_and_escapes_newlines(qa_env: Path) -> None:
    answer = "첫 줄\n[2026-01-01 00:00:00] INFO - 위장 로그\n둘째 줄"
    qa_trace.record_qa_turn(
        source="astream", session_id="s", query="질문", answer=answer
    )
    qa_trace.reset_for_tests()

    lines = qa_env.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["answer"] == answer
    assert record["query"] == "질문"


def test_max_answer_chars_truncates(
    qa_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "QA_LOG_MAX_ANSWER_CHARS", 5)
    qa_trace.record_qa_turn(
        source="aquery", session_id="s", query="q", answer="0123456789"
    )
    qa_trace.reset_for_tests()
    records = _read_records(qa_env)
    assert records[0]["answer"] == "01234"


def test_rotation_creates_backup(qa_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "QA_LOG_ROTATE_MAX_BYTES", 10)
    monkeypatch.setattr(cfg, "QA_LOG_ROTATE_BACKUPS", 1)
    for index in range(2):
        qa_trace.record_qa_turn(
            source="aquery", session_id="s", query=f"q{index}", answer="a" * 50
        )
    qa_trace.reset_for_tests()

    backup = qa_env.with_name(qa_env.name + ".1")
    assert backup.exists()
    assert qa_env.exists()
    assert _read_records(backup)[0]["query"] == "q0"
    assert _read_records(qa_env)[0]["query"] == "q1"


async def _drain(gen: AsyncIterator[Any]) -> None:
    while True:
        try:
            await anext(gen)
        except StopAsyncIteration:
            return


def _astream_events(*, is_cached: bool, answers: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "event": "on_chain_stream",
            "metadata": {},
            "data": {
                "chunk": {"preprocess": {"intent": "rag", "is_cached": is_cached}}
            },
        },
        {
            "event": "on_chain_stream",
            "metadata": {},
            "data": {
                "chunk": {
                    "retrieve": {
                        "relevant_docs": [
                            Document(page_content="x", metadata={"stable_id": "c1"})
                        ]
                    }
                }
            },
        },
    ]
    for answer in answers:
        events.append(
            {
                "event": "on_chain_stream",
                "metadata": {},
                "data": {
                    "chunk": {
                        "generate": {
                            "response": answer,
                            "citations": [{"doc_id": "c1"}],
                            "confidence": 0.8,
                            "prompt_version": "v1",
                            "regenerated": "",
                            "parse_failed": False,
                            "echo_detected": False,
                            "over_budget": False,
                            "performance": {"input_token_count": 3},
                        }
                    }
                },
            }
        )
    return events


@pytest.mark.asyncio
async def test_astream_records_last_generate_once() -> None:
    rag = RAGSystem(session_id="qa-hook")
    engine = AsyncMock()
    breaker = MagicMock()
    registry = MagicMock()
    registry.get_breaker.return_value = breaker
    manager = MagicMock()
    session_manager = MagicMock()
    session_manager.get.return_value = None
    events = _astream_events(is_cached=True, answers=["중간", "최종"])

    async def fake_stream(*args: Any, **kwargs: Any):
        for event in events:
            yield event

    breaker.call_async_stream = fake_stream

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
        stack.enter_context(patch("core.rag_core.hydrate_documents", AsyncMock()))
        record_mock = stack.enter_context(patch("core.rag_core.record_qa_turn"))
        gen = await rag.astream("질문")
        await _drain(gen)

    record_mock.assert_called_once()
    kwargs = record_mock.call_args.kwargs
    assert kwargs["source"] == "astream"
    assert kwargs["answer"] == "최종"
    assert kwargs["is_cached"] is True
    assert kwargs["intent"] == "rag"
    assert kwargs["relevant_docs"][0].metadata["stable_id"] == "c1"


@pytest.mark.asyncio
async def test_aquery_records_once() -> None:
    rag = RAGSystem(session_id="qa-hook-query")
    engine = MagicMock()
    engine.ainvoke = AsyncMock(
        return_value={
            "response": "답변입니다",
            "relevant_docs": [],
            "performance": {"output_token_count": 5},
            "is_cached": True,
            "intent": "general",
            "citations": [],
            "confidence": 0.7,
            "prompt_version": "v1",
        }
    )
    session_manager = MagicMock()
    session_manager.get.return_value = None
    monitor = MagicMock()

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
        stack.enter_context(patch("core.rag_core.hydrate_documents", AsyncMock()))
        stack.enter_context(
            patch("core.rag_core.get_resource_manager", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "services.monitoring.performance_monitor.get_performance_monitor",
                return_value=monitor,
            )
        )
        stack.enter_context(
            patch("services.monitoring.performance_monitor.OperationType", MagicMock())
        )
        stack.enter_context(
            patch("core.graph.graph_builder.format_context", return_value="ctx")
        )
        record_mock = stack.enter_context(patch("core.rag_core.record_qa_turn"))
        result = await rag.aquery("질문")

    assert result["response"] == "답변입니다"
    record_mock.assert_called_once()
    kwargs = record_mock.call_args.kwargs
    assert kwargs["answer"] == "답변입니다"
    assert kwargs["is_cached"] is True
    assert kwargs["intent"] == "general"
