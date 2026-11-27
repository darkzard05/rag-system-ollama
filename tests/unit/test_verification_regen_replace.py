"""Verification regeneration must replace the first streamed answer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from api.schemas import AnswerStructure, GraphState
from core.graph._generate import generate


@pytest.mark.asyncio
async def test_verification_regeneration_emits_answer_replace() -> None:
    state = GraphState(
        input="질문",
        relevant_docs=[Document(page_content="문서 내용")],
        verification_issues=["이전 답변에 근거가 없습니다"],
        regeneration_count=1,
    )
    config: RunnableConfig = {"configurable": {"llm": MagicMock()}}
    writer = AsyncMock()
    llm_result = {
        "full_response": '{"final_answer": "수정 답변", "citations": []}',
        "full_thought": "",
        "last_metadata": {},
        "ttft_ms": 10.0,
        "first_token_ms": 10.0,
        "generate_ms": 50.0,
        "parsed_answer": AnswerStructure(
            final_answer="수정 답변", citations=[], confidence=0.9
        ),
        "parse_failed": False,
        "truncated": False,
    }

    with (
        patch(
            "core.graph._generate._generate_pass",
            new_callable=AsyncMock,
            return_value=llm_result,
        ),
        patch(
            "core.graph._generate._dispatch_event", new_callable=AsyncMock
        ) as dispatch,
    ):
        result = await generate(state, config, writer=writer)

    assert result["response"] == "수정 답변"
    replace_calls = [
        call for call in dispatch.await_args_list if call.args[0] == "answer_replace"
    ]
    assert len(replace_calls) == 1
    assert replace_calls[0].args[1]["content"] == "수정 답변"
