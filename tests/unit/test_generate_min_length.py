"""Minimum-answer-length regeneration tests (mocked LLM, hermetic).

The 4B model tends to ignore the prompt's 800~1,200자 target and returns
400~600자 answers. ``generation_validation.regen_on_short`` adds a single
expansion pass for substantive document-grounded answers, while greetings and
refusals stay short.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph._graph_core import _extract_json_string, _extract_partial_answer
from core.graph.graph_builder import generate
from core.model_loader import ModelManager

_NUM_CTX = 8192
_NUM_PREDICT = 2048

_SHORT_ANSWER = "짧은 답변입니다. " * 20
_LONG_ANSWER = "충분히 긴 답변입니다. " * 80

_VALIDATION = {
    "over_budget_chars": 1400,
    "regen_floor_chars": 700,
    "min_answer_chars": 800,
    "regen_on_invalid": False,
    "regen_on_over_budget": False,
    "regen_on_short": True,
}


def _make_docs() -> list[Document]:
    return [
        Document(
            page_content="문서 근거 내용입니다.",
            metadata={"rerank_score": 0.9, "page": 1},
        ),
        Document(
            page_content="추가 근거 내용입니다.",
            metadata={"rerank_score": 0.8, "page": 2},
        ),
    ]


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


def _payload(answer: str) -> str:
    return json.dumps(
        {"final_answer": answer, "citations": [], "confidence": 0.9},
        ensure_ascii=False,
    )


async def _run_generate(
    answers: list[str],
    query: str = "문서의 핵심 기여를 설명해줘.",
    metadatas: list[dict] | None = None,
    *,
    raw_payloads: bool = False,
) -> tuple[dict, list[list], AsyncMock]:
    sent_messages: list[list] = []
    calls = {"n": 0}
    mock_dispatch = AsyncMock()

    async def mock_astream(messages, config=None):  # noqa: ARG001
        sent_messages.append(messages)
        index = min(calls["n"], len(answers) - 1)
        calls["n"] += 1
        metadata: dict = {"prompt_eval_count": 5}
        if metadatas:
            metadata.update(metadatas[index])
        content = answers[index] if raw_payloads else _payload(answers[index])
        yield AIMessageChunk(content=content, response_metadata=metadata)

    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_llm
    mock_llm.astream = mock_astream
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    state = {"input": query, "relevant_docs": _make_docs(), "is_cached": False}
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch("core.graph._generate.OLLAMA_NUM_CTX", _NUM_CTX),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", _NUM_PREDICT),
        patch(
            "core.graph._generate.GENERATE_PROMPT_CONFIG",
            {"max_context_tokens": 6000, "min_docs_keep": 2},
        ),
        patch(
            "core.graph._generate.GENERATION_VALIDATION_CONFIG",
            _VALIDATION,
        ),
        patch("core.graph._generate.count_tokens_rough", return_value=100),
        patch("core.graph._glue.adispatch_custom_event", new=mock_dispatch),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())
    return result, sent_messages, mock_dispatch


@pytest.mark.asyncio
async def test_generate_regenerates_when_answer_too_short():
    """문서 근거가 있는데 답변이 800자 미만이면 1회 확장 재생성한다."""
    result, sent_messages, dispatch = await _run_generate([_SHORT_ANSWER, _LONG_ANSWER])

    assert len(sent_messages) == 2
    assert len(result["response"]) >= 800
    assert result["regenerated"] == "under_budget"
    corrective_human = sent_messages[1][1].content
    assert "목표 최소" in corrective_human
    assert "citations[]" in corrective_human

    replace_calls = [
        call for call in dispatch.await_args_list if call.args[0] == "answer_replace"
    ]
    assert len(replace_calls) == 1
    assert replace_calls[0].args[1]["content"] == result["response"]


@pytest.mark.asyncio
async def test_generate_skips_short_regen_for_greeting():
    """인사/잡담 질문은 짧은 답변이 정상이므로 재생성하지 않는다."""
    greeting = "안녕하세요! 저는 문서 분석을 돕는 AI 어시스턴트입니다. " * 10
    result, sent_messages, dispatch = await _run_generate(
        [greeting], query="안녕, 너는 누구니?"
    )

    assert len(sent_messages) == 1
    assert result["response"] == greeting
    assert not any(
        call.args[0] == "answer_replace" for call in dispatch.await_args_list
    )


@pytest.mark.asyncio
async def test_generate_skips_short_regen_for_refusal():
    """근거 없음 거절 답변은 짧아도 정상이므로 재생성하지 않는다."""
    refusal = "제공된 문서에서 해당 정보를 확인할 수 없습니다. " * 8
    result, sent_messages, dispatch = await _run_generate([refusal])

    assert len(sent_messages) == 1
    assert result["response"] == refusal
    assert not any(
        call.args[0] == "answer_replace" for call in dispatch.await_args_list
    )


def test_extract_partial_answer_recovers_truncated_value():
    """num_predict 절단으로 닫는 따옴표가 없어도 본문을 복구한다."""
    truncated = '{"final_answer": "잘린 답변 본문입니다. citations는 뒤에'
    assert (
        _extract_partial_answer(truncated, allow_eof=True)
        == "잘린 답변 본문입니다. citations는 뒤에"
    )
    # 절단이 확인되지 않은 깨진 JSON 은 fail-closed 로 복구하지 않는다.
    assert _extract_partial_answer(truncated) is None


def test_extract_json_string_requires_closing_quote_by_default():
    """기본 동작(닫는 따옴표 필수)은 유지해 잘린 anchor 채택을 막는다."""
    truncated = '{"anchor": "잘린 anchor'
    start = truncated.index(":", truncated.index('"anchor"')) + 1
    assert _extract_json_string(truncated, start) == (None, start)
    assert _extract_json_string(truncated, start, allow_eof=True)[0] == "잘린 anchor"


@pytest.mark.asyncio
async def test_generate_marks_truncated_single_pass():
    """1차 패스가 num_predict에서 절단되면 부분 복구 + truncated 플래그를 남긴다."""
    body = "절단된 긴 답변 본문입니다. " * 59 + "끝까지"
    result, sent_messages, _ = await _run_generate(
        [f'{{"final_answer": "{body}'],
        metadatas=[{"done_reason": "length", "eval_count": _NUM_PREDICT}],
        raw_payloads=True,
    )

    assert len(sent_messages) == 1
    assert result["response"] == body
    assert result["parse_failed"] is True
    assert result["performance"]["truncated"] is True
    assert result["regenerated"] == ""


@pytest.mark.asyncio
async def test_generate_keeps_first_pass_when_regen_truncated():
    """재생성 패스가 절단되면 정상 1차 패스를 유지한다 (답변 유실 방지)."""
    first = "정상적인 1차 답변입니다. " * 20
    truncated_second = '{"final_answer": "재생성 답변이 길어지다가 절단'
    result, sent_messages, dispatch = await _run_generate(
        [_payload(first), truncated_second],
        metadatas=[{}, {"done_reason": "length", "eval_count": _NUM_PREDICT}],
        raw_payloads=True,
    )

    assert len(sent_messages) == 2
    assert result["response"] == first
    assert result["parse_failed"] is False
    assert result["regenerated"] == ""
    replace_calls = [
        call for call in dispatch.await_args_list if call.args[0] == "answer_replace"
    ]
    assert len(replace_calls) == 1
    assert replace_calls[0].args[1]["content"] == first


@pytest.mark.asyncio
async def test_generate_keeps_first_pass_when_regen_is_refusal():
    """재생성 결과가 거절 폴백이면 정상 1차 패스를 유지한다."""
    first = "정상적인 1차 답변입니다. " * 20
    refusal = "제공된 문서에서 확인할 수 없습니다."
    result, sent_messages, _ = await _run_generate([first, refusal])

    assert len(sent_messages) == 2
    assert result["response"] == first
    assert result["regenerated"] == ""
