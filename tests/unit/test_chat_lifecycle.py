import asyncio
import pytest
from unittest.mock import MagicMock, patch

from src.core.session import SessionManager
from src.ui.components.chat import _process_streaming


@pytest.mark.asyncio
async def test_chat_streaming_lifecycle():
    # Setup
    session_id = "test_session"
    query = "Test query"
    model_name = "test_model"

    # Mock RAGSystem.astream to return a mock event generator
    mock_event = MagicMock()
    mock_event.status = "Thinking..."
    mock_event.metadata = {"documents": [MagicMock()]}
    mock_event.performance = {"total_time": 1.0, "tps": 30.0}
    mock_event.thought = "I am thinking"
    mock_event.content = "Hello world"

    async def mock_astream(q, model_name=None):
        yield (
            "custom",
            {
                "content": "Hello world",
                "thought": "I am thinking",
                "status": "Thinking...",
            },
        )
        yield (
            "updates",
            {"generate": {"performance": {"token_count": 10, "input_token_count": 5}}},
        )

        # Mock Streamlit components
        with (
            patch("streamlit.chat_message"),
            patch("streamlit.empty"),
            patch("streamlit.markdown"),
            patch("src.core.rag_core.RAGSystem.astream", side_effect=mock_astream),
            patch("api.streaming_handler.get_streaming_handler") as mock_handler_getter,
        ):
            # Mock streaming handler
            mock_handler = MagicMock()

            async def mock_stream_graph_events(gen):
                async for event in gen:
                    yield event

            mock_handler.stream_graph_events.side_effect = mock_stream_graph_events
            mock_handler_getter.return_value = mock_handler

        # Initial state
        SessionManager.reset_all_state()
        SessionManager.set("is_generating_answer", False, session_id)
        SessionManager.set("file_hash", "mock_hash", session_id=session_id)

        # Execute
        await _process_streaming(query, model_name, session_id)

        # Verify state transitions
        # 1. is_generating_answer should be False after completion (finally block)
        assert (
            SessionManager.get("is_generating_answer", session_id=session_id) is False
        )

        # 2. Assistant message should be added to history
        messages = SessionManager.get_messages(session_id=session_id)
        assert len(messages) > 0
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hello world"
        assert messages[-1]["thought"] == "I am thinking"


if __name__ == "__main__":
    asyncio.run(test_chat_streaming_lifecycle())
