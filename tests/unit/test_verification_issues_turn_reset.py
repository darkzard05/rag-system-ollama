"""Stale verification failures must not leak into a later turn's prompt."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph.graph_builder import generate, preprocess


class MockWriter:
    """Stand-in for the LangGraph stream writer."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


async def _capture_prompt(state: dict) -> str:
    sent: list = []
    mock_llm = MagicMock()
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    async def mock_astream(messages, config=None):  # noqa: ARG001
        sent.append(messages)
        yield AIMessageChunk(content='{"final_answer": "답변"}', response_metadata={})

    mock_llm.bind.return_value.astream = mock_astream
    config = {"configurable": {"llm": mock_llm}}
    with (
        patch("core.graph._glue.adispatch_custom_event", new=AsyncMock()),
        patch("core.graph._generate.count_tokens_rough", return_value=1),
    ):
        await generate(state, config, writer=MockWriter())

    assert sent
    return str(sent[0][0].content)


@pytest.mark.asyncio
async def test_new_turn_prompt_has_no_prior_failure_reason() -> None:
    """preprocess clears verification_issues before the next generate call."""
    prior_turn_end = {
        "input": "이전 질문",
        "verification_issues": ["유효하지 않은 인용: [doc:x]"],
        "relevant_docs": [
            Document(page_content="문서 본문", metadata={"rerank_score": 0.9})
        ],
        "is_cached": False,
    }

    next_turn_out = await preprocess(
        {"input": "새 질문입니다"}, {"configurable": {}}, writer=MockWriter()
    )
    merged = {**prior_turn_end, **next_turn_out}
    merged["input"] = "새 질문입니다"
    merged["relevant_docs"] = prior_turn_end["relevant_docs"]

    assert "verification_issues" in next_turn_out
    assert next_turn_out["verification_issues"] is None
    prompt = await _capture_prompt(merged)
    assert "검증 실패했습니다" not in prompt
