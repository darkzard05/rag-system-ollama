"""
Todo 9 검증 (R4-08): thought/content 분리 미지원 LLM의 폴백 분기에서
`chunk.content`가 리스트(복합 콘텐츠)여도 TypeError 없이 텍스트로 병합되어야 합니다.

기존 결함: 폴백에서 `content_chunk = chunk.content`(리스트 가능)를 그대로 두고
`full_response += content_chunk`(str+list)로 연결해
`TypeError: can only concatenate str (not "list") to str`이 발생했다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from core.graph_builder import generate
from core.model_loader import ModelManager


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_generate_fallback_merges_list_content():
    """hasattr(_convert_chunk_to_thought_and_content) == False + content 리스트여도 정상 병합."""
    mock_llm = MagicMock()
    # thought 분리 미지원 LLM 시뮬레이션 — MagicMock은 접근 시 자동 생성하므로 명시 제거
    if hasattr(mock_llm, "_convert_chunk_to_thought_and_content"):
        del mock_llm._convert_chunk_to_thought_and_content

    async def mock_astream(messages, config=None):  # noqa: ARG001
        yield AIMessageChunk(
            content=[
                {"type": "text", "text": "리스트 "},
                {"type": "text", "text": "병합 답변"},
            ],
            response_metadata={"prompt_eval_count": 3},
        )

    mock_llm.astream = mock_astream

    state = {
        "input": "질문",
        "relevant_docs": [MagicMock(page_content="참고 문서", metadata={"page": 1})],
        "is_cached": False,
    }
    config = {"configurable": {"llm": mock_llm}}

    # 폴백 분기 진입 조건 확인
    assert not hasattr(mock_llm, "_convert_chunk_to_thought_and_content")

    with (
        patch("core.graph_builder.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())

    assert isinstance(result["response"], str)
    assert result["response"] == "리스트 병합 답변"


@pytest.mark.asyncio
async def test_generate_fallback_mixed_content_blocks():
    """리스트에 text 블록뿐 아니라 other/reasoning 블록이 섞여도 크래시 없이 text만 병합합니다."""
    mock_llm = MagicMock()
    if hasattr(mock_llm, "_convert_chunk_to_thought_and_content"):
        del mock_llm._convert_chunk_to_thought_and_content

    async def mock_astream(messages, config=None):  # noqa: ARG001
        yield AIMessageChunk(
            content=[
                {"type": "reasoning", "reasoning": "생각 중..."},
                {"type": "text", "text": "최종 답변"},
            ],
            response_metadata={"prompt_eval_count": 3},
        )

    mock_llm.astream = mock_astream

    state = {
        "input": "질문",
        "relevant_docs": [],
        "intent": "general",
        "is_cached": False,
    }
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch("core.graph_builder.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())

    assert result["response"] == "최종 답변"
