"""B8: verification_issues → 재생성 프롬프트 주입 단위 테스트.

verify → regenerate 재시도가 맹목 재실행이 되지 않도록, generate 노드는 상태의
verification_issues를 프롬프트에 주입한다. 사유가 없으면(정상 경로) 프롬프트가
byte-identical로 유지되어야 한다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph.graph_builder import generate
from core.model_loader import ModelManager

_NUM_CTX = 8192
_NUM_PREDICT = 2048


def _make_docs() -> list[Document]:
    return [
        Document(
            page_content="검색된 문서 본문 내용",
            metadata={"rerank_score": 0.9, "page": 1},
        ),
    ]


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


async def _run_generate(state: dict, sent_messages: list) -> dict:
    mock_llm = MagicMock()
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    async def mock_astream(messages, config=None):  # noqa: ARG001
        sent_messages.append(messages)
        yield AIMessageChunk(
            content='{"final_answer": "답변", "reasoning": "추론"}',
            response_metadata={"prompt_eval_count": 5},
        )

    mock_llm.bind.return_value.astream = mock_astream
    config = {"configurable": {"llm": mock_llm}}

    def fake_count(text: str) -> int:  # noqa: ARG001
        return 1

    with (
        patch("core.graph._generate.OLLAMA_NUM_CTX", _NUM_CTX),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", _NUM_PREDICT),
        patch("core.graph._generate.count_tokens_rough", side_effect=fake_count),
        patch("core.graph._glue.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        return await generate(state, config, writer=MagicMock())


def _base_state() -> dict:
    return {"input": "질문입니다", "relevant_docs": _make_docs(), "is_cached": False}


@pytest.mark.asyncio
async def test_generate_prompt_includes_verification_issues_when_present():
    """verification_issues 존재 시 재생성 프롬프트에 사유가 주입됩니다."""
    sent_messages: list = []
    state = _base_state()
    state["verification_issues"] = ["유효하지 않은 인용: [doc:x]"]
    await _run_generate(state, sent_messages)

    assert sent_messages, "LLM이 호출되어야 합니다"
    prompt_text = sent_messages[0][0].content
    assert "유효하지 않은 인용: [doc:x]" in prompt_text
    assert "검증 실패했습니다" in prompt_text


@pytest.mark.asyncio
async def test_generate_prompt_unchanged_without_issues():
    """사유가 없으면 정상 경로 프롬프트는 인젝션 구간 없이 동일합니다."""
    sent_messages: list = []
    await _run_generate(_base_state(), sent_messages)

    assert sent_messages, "LLM이 호출되어야 합니다"
    prompt_text = sent_messages[0][0].content
    assert "검증 실패했습니다" not in prompt_text
