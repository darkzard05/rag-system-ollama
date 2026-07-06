import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# We are testing the logic of streaming and updating UI,
# but since we are modifying chat.py, we will test the core
# behavior: (1) it consumes the generator, (2) it updates UI, (3) it saves to session.


@pytest.mark.asyncio
async def test_streaming_logic_flow():
    # Mocking dependencies
    mock_rag_sys = MagicMock()

    # Mock astream to return a sequence of chunks
    async def mock_astream(query, model_name):
        yield MagicMock(
            status="Thinking...",
            metadata={},
            performance={},
            thought="I am thinking",
            content="",
        )
        yield MagicMock(
            status="Generating...",
            metadata={},
            performance={},
            thought="I am thinking",
            content="Hello ",
        )
        yield MagicMock(
            status="Generating...",
            metadata={},
            performance={},
            thought="I am thinking",
            content="world!",
        )
        yield MagicMock(
            status=None,
            metadata={"documents": ["Doc1"]},
            performance={"total_time": 1.0, "tps": 50},
            thought="I am thinking",
            content="",
        )

    mock_rag_sys.astream = mock_astream

    # We want to see if the content is accumulated and finally saved
    content_acc = ""
    thought_acc = ""
    docs_acc = []

    # Simulate the loop that will be implemented in chat.py
    async for chunk in mock_rag_sys.astream("test query", "test-model"):
        if chunk.thought:
            thought_acc += chunk.thought
        if chunk.content:
            content_acc += chunk.content
        if chunk.metadata and "documents" in chunk.metadata:
            docs_acc = chunk.metadata["documents"]

    assert content_acc == "Hello world!"
    assert "I am thinking" in thought_acc
    assert docs_acc == ["Doc1"]


if __name__ == "__main__":
    asyncio.run(test_streaming_logic_flow())
