from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from api.schemas import AnswerStructure, GraphState
from core.graph._generate import generate
from ui.components.chat import _resolve_final_text
from ui.components.streaming_core import _content_generator, _AUX_STATE_KEY
from ui.components.streaming import consume_stream_into_message
from core.session import SessionManager
from api.stream_pipeline import StreamChunk


def test_resolve_final_text_normal():
    text, replaced = _resolve_final_text("정상 답변", "정상 답변", None)
    assert text == "정상 답변"
    assert replaced is False


def test_resolve_final_text_empty_streamed():
    text, replaced = _resolve_final_text("폴백 답변", "", None)
    assert text == "폴백 답변"
    assert replaced is True


@pytest.mark.asyncio
async def test_generate_single_pass_parse_failed_mock_dispatch():
    state = GraphState(
        input="질문",
        relevant_docs=[Document(page_content="문서 내용")],
    )
    config: RunnableConfig = {"configurable": {"llm": MagicMock()}}
    writer = AsyncMock()

    mock_llm_result = {
        "full_response": "일반 텍스트 답변",
        "full_thought": "",
        "last_metadata": {},
        "ttft_ms": 10.0,
        "first_token_ms": 10.0,
        "generate_ms": 50.0,
        "parsed_answer": AnswerStructure(
            final_answer="일반 텍스트 답변", citations=[], confidence=0.0
        ),
        "parse_failed": True,
        "truncated": False,
    }

    with (
        patch(
            "core.graph._generate._generate_pass", new_callable=AsyncMock
        ) as mock_pass,
        patch(
            "core.graph._generate._dispatch_event", new_callable=AsyncMock
        ) as mock_dispatch,
    ):
        mock_pass.return_value = mock_llm_result
        result = await generate(state, config, writer=writer)

        assert result["response"] == "일반 텍스트 답변"
        event_names = [
            call.args[0] for call in mock_dispatch.call_args_list if len(call.args) > 0
        ]
        assert "answer_replace" in event_names


@pytest.mark.asyncio
async def test_content_generator_is_replace_resets_extractor():
    sid = "test-sid"
    msg_id = "test-msg"
    SessionManager.set_session_id(sid)

    chunks = [
        StreamChunk(content='{"final_answer": "부분\\', raw_json=True),
        StreamChunk(content="확정 답변", is_replace=True, is_final=True),
    ]

    with patch("ui.components.streaming_core.stream_chunks", return_value=iter(chunks)):
        gen = _content_generator("q", "model", sid, msg_id)
        list(gen)
        aux = SessionManager.get(_AUX_STATE_KEY, {}, sid)
        assert aux.get("content") == "확정 답변"


def test_consume_stream_into_message_is_replace_resets_extractor():
    sid = "test-sid-2"
    msg_id = "test-msg-2"
    SessionManager.set_session_id(sid)

    chunks = [
        StreamChunk(content="비구조 텍스트", raw_json=False),
        StreamChunk(content="확정 답변", is_replace=True, is_final=True),
    ]

    with patch("ui.components.streaming.stream_chunks", return_value=iter(chunks)):
        msg = consume_stream_into_message(sid, "q", "model", msg_id=msg_id)
        assert msg is not None
        assert msg.get("content") == "확정 답변"
