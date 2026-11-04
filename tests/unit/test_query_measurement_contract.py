"""QueryMeasurement contract tests shared by the UI and eval harness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

import services.query_measurement as query_measurement
from api.stream_pipeline import StreamChunk
from common.final_answer import (
    FinalAnswerExtractor,
    extract_final_answer_delta,
    recover_final_answer,
)
from services.query_measurement import (
    QUERY_CONTRACT_VERSION,
    QueryMeasurement,
    run_query_measurement,
)
from ui.components.streaming import (
    _extract_final_answer_delta,
    _FinalAnswerExtractor,
    _recover_final_answer,
)


class FakeRAG:
    async def astream(
        self, query: str, model_name: str | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        async def events() -> AsyncIterator[tuple[str, Any]]:
            yield ("custom", {"query": query})

        return events()


class FakeHandler:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = chunks

    async def stream_graph_events(
        self, event_stream: AsyncIterator[tuple[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self._chunks:
            yield chunk


def _install_handler(
    monkeypatch: pytest.MonkeyPatch, chunks: list[StreamChunk]
) -> None:
    monkeypatch.setattr(
        query_measurement,
        "get_streaming_handler",
        lambda: FakeHandler(chunks),
    )


def test_ui_and_eval_share_the_same_extractor_symbols() -> None:
    assert _FinalAnswerExtractor is FinalAnswerExtractor
    assert _extract_final_answer_delta is extract_final_answer_delta
    assert _recover_final_answer is recover_final_answer


async def test_structured_stream_separates_display_and_raw_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_handler(
        monkeypatch,
        [
            StreamChunk(
                content='{"reasoning":"r","final_answer":"안',
                raw_json=True,
            ),
            StreamChunk(content="녕", raw_json=True),
            StreamChunk(content='하세요","citations":[]}', raw_json=True),
            StreamChunk(metadata={"documents": [{"page": 1}]}),
            StreamChunk(
                performance={
                    "eval_count": 4,
                    "parse_failed": True,
                    "echo_detected": False,
                    "over_budget": True,
                }
            ),
        ],
    )

    measured = await run_query_measurement(FakeRAG(), "질문")

    assert measured.contract_version == QUERY_CONTRACT_VERSION
    assert measured.display_answer == "안녕하세요"
    assert measured.raw_answer == (
        '{"reasoning":"r","final_answer":"안녕하세요","citations":[]}'
    )
    assert measured.answer_chars == 5
    assert measured.raw_answer_chars == len(measured.raw_answer)
    assert measured.flags.raw_json_seen is True
    assert measured.flags.display_extraction_failed is False
    assert measured.eval_count == 4
    assert measured.parse_failed is True
    assert measured.over_budget is True
    assert measured.documents == [{"page": 1}]
    assert measured.timing.ttft_s is not None
    assert measured.timing.ttft_e2e_s is not None
    assert measured.timing.ttft_display_s is not None
    assert measured.timing.ttft_e2e_s >= measured.timing.ttft_s


async def test_raw_json_without_final_answer_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_handler(
        monkeypatch,
        [StreamChunk(content='{"reasoning":"only"}', raw_json=True)],
    )

    measured = await run_query_measurement(FakeRAG(), "질문")

    assert measured.display_answer == ""
    assert measured.raw_answer == '{"reasoning":"only"}'
    assert measured.flags.raw_json_seen is True
    assert measured.flags.display_extraction_failed is True


async def test_retrieval_only_stops_after_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = {"page": 7}
    _install_handler(
        monkeypatch,
        [
            StreamChunk(metadata={"documents": [doc]}),
            StreamChunk(content='{"final_answer":"not consumed"}', raw_json=True),
        ],
    )

    measured = await run_query_measurement(FakeRAG(), "질문", retrieval_only=True)

    assert measured.documents == [doc]
    assert measured.display_answer == ""
    assert measured.raw_answer == ""
    assert measured.flags.retrieval_only is True
    assert measured.flags.eval_count_missing is True


def test_measurement_properties_handle_missing_eval_count() -> None:
    measured = QueryMeasurement(
        contract_version=QUERY_CONTRACT_VERSION,
        display_answer="답",
        raw_answer="",
        documents=[],
        performance={},
        timing=query_measurement.QueryTiming(
            setup_s=0.1,
            ttft_s=None,
            ttft_e2e_s=None,
            ttft_display_s=None,
            ttft_display_e2e_s=None,
            generation_s=None,
            total_s=0.2,
        ),
        flags=query_measurement.QueryFlags(
            retrieval_only=False,
            raw_json_seen=False,
            display_extraction_failed=False,
            eval_count_missing=True,
        ),
    )

    assert measured.eval_count is None
    assert measured.tps is None
    assert measured.parse_failed is False
