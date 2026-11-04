"""Eval harness must consume QueryMeasurement instead of raw StreamChunks."""

from __future__ import annotations

from typing import Any, cast

import pytest
import scripts.evaluation.eval_quality as eval_quality

from services.query_measurement import (
    QUERY_CONTRACT_VERSION,
    QueryFlags,
    QueryMeasurement,
    QueryTiming,
)


def _measurement(**overrides: Any) -> QueryMeasurement:
    values: dict[str, Any] = {
        "contract_version": QUERY_CONTRACT_VERSION,
        "display_answer": "표시 답변",
        "raw_answer": '{"final_answer":"표시 답변"}',
        "documents": [],
        "performance": {"eval_count": 12},
        "timing": QueryTiming(
            setup_s=0.5,
            ttft_s=0.25,
            ttft_e2e_s=0.75,
            ttft_display_s=0.3,
            ttft_display_e2e_s=0.8,
            generation_s=2.0,
            total_s=3.0,
        ),
        "flags": QueryFlags(
            retrieval_only=False,
            raw_json_seen=True,
            display_extraction_failed=False,
            eval_count_missing=False,
        ),
    }
    values.update(overrides)
    return QueryMeasurement(**values)


async def test_consume_stream_delegates_to_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _measurement()
    calls: dict[str, Any] = {}

    async def fake_run(
        rag: Any,
        query: str,
        model_name: str | None = None,
        *,
        retrieval_only: bool = False,
    ) -> QueryMeasurement:
        calls.update(
            {
                "rag": rag,
                "query": query,
                "model_name": model_name,
                "retrieval_only": retrieval_only,
            }
        )
        return expected

    monkeypatch.setattr(eval_quality, "run_query_measurement", fake_run)
    rag = cast(Any, object())

    result = await eval_quality._consume_stream(rag, "질문", no_llm=True)

    assert result is expected
    assert calls == {
        "rag": rag,
        "query": "질문",
        "model_name": eval_quality.DEFAULT_OLLAMA_MODEL,
        "retrieval_only": True,
    }


async def test_consume_stream_rejects_contract_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run(
        rag: Any,
        query: str,
        model_name: str | None = None,
        *,
        retrieval_only: bool = False,
    ) -> QueryMeasurement:
        return _measurement(contract_version=QUERY_CONTRACT_VERSION + 1)

    monkeypatch.setattr(eval_quality, "run_query_measurement", fake_run)

    with pytest.raises(RuntimeError, match="contract mismatch"):
        await eval_quality._consume_stream(cast(Any, object()), "질문", no_llm=False)


def test_assemble_measurement_uses_display_answer_and_contract_timings() -> None:
    question = eval_quality.QuestionSpec(
        source="testset",
        row=1,
        query="질문",
        ground_truth="정답",
    )
    record = eval_quality._empty_record(question, None, skipped=False)

    eval_quality._assemble_measurement(record, question, _measurement())

    assert record["answer"] == "표시 답변"
    assert record["raw_answer"] == '{"final_answer":"표시 답변"}'
    assert record["answer_chars"] == len("표시 답변")
    assert record["raw_answer_chars"] == len(record["raw_answer"])
    assert record["display_extraction_failed"] == 0
    assert record["eval_count"] == 12
    assert record["ttft"] == 0.25
    assert record["ttft_setup"] == 0.5
    assert record["ttft_e2e"] == 0.75
    assert record["ttft_display"] == 0.3
    assert record["ttft_display_e2e"] == 0.8
    assert record["tps"] == 6.0


def test_assemble_measurement_marks_display_extraction_failure() -> None:
    question = eval_quality.QuestionSpec(
        source="testset",
        row=1,
        query="질문",
        ground_truth="정답",
    )
    record = eval_quality._empty_record(question, None, skipped=False)
    measured = _measurement(
        display_answer="",
        flags=QueryFlags(
            retrieval_only=False,
            raw_json_seen=True,
            display_extraction_failed=True,
            eval_count_missing=False,
        ),
    )

    eval_quality._assemble_measurement(record, question, measured)

    assert record["answer"] == ""
    assert record["display_extraction_failed"] == 1
