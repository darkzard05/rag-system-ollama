"""The generate prompt must inject context and query as bare text."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph.graph_builder import generate
from core.model_loader import ModelManager

_CONTEXT_MARKER = "MARKER_CONTEXT_UNIQUE"
_QUERY_MARKER = "MARKER_QUERY_UNIQUE"


@pytest.mark.asyncio
async def test_context_and_query_are_not_wrapped_in_braces() -> None:
    """A model reading the prompt sees the context, not a braced literal."""
    sent_messages: list[list] = []

    async def mock_astream(messages, config=None):  # noqa: ARG001
        sent_messages.append(messages)
        payload = json.dumps(
            {"final_answer": "답변", "citations": [], "confidence": 0.9},
            ensure_ascii=False,
        )
        yield AIMessageChunk(content=payload, response_metadata={})

    llm = MagicMock()
    llm.bind.return_value = llm
    llm.astream = mock_astream
    llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    doc = Document(
        page_content=_CONTEXT_MARKER,
        metadata={"rerank_score": 0.9, "page": 1},
    )
    state = {"input": _QUERY_MARKER, "relevant_docs": [doc], "is_cached": False}
    config = {"configurable": {"llm": llm}}
    session = MagicMock()
    session.return_value.__aenter__ = AsyncMock()
    session.return_value.__aexit__ = AsyncMock()

    with (
        patch("core.graph._generate.GENERATION_VALIDATION_CONFIG", {}),
        patch("core.graph._generate.count_tokens_rough", return_value=10),
        patch("core.graph._glue.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", session),
    ):
        await generate(state, config, writer=MagicMock())

    assert sent_messages, "generate must call the LLM"
    system_prompt = sent_messages[0][0].content
    assert _CONTEXT_MARKER in system_prompt
    assert _QUERY_MARKER in system_prompt
    assert "{" + _CONTEXT_MARKER + "}" not in system_prompt
    assert "{" + _QUERY_MARKER + "}" not in system_prompt
