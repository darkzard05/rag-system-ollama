import pytest
import pytest_asyncio
from langchain_core.messages import AIMessageChunk
from api.stream_pipeline import StreamingResponseHandler


# 모의 비동기 스트림 생성기
async def mock_event_stream():
    yield ("custom", {"status": "test_status"})
    yield ("messages", AIMessageChunk(content="test_content"))


@pytest.mark.asyncio
async def test_streaming_response_handler_modes():
    handler = StreamingResponseHandler()

    # 모의 스트림 전달
    stream = mock_event_stream()

    chunks = []
    async for chunk in handler.stream_graph_events(stream):
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert chunks[0].status == "test_status"
    assert chunks[1].content == "test_content"


@pytest.mark.asyncio
async def test_stream_graph_events_aclose_runtime_error():
    from api.stream_pipeline import StreamingResponseHandler

    async def dummy_events():
        yield ("custom", {"content": "hello"})
        yield ("custom", {"content": "world"})

    handler = StreamingResponseHandler()
    gen = handler.stream_graph_events(dummy_events(), _remaining=None)

    item = await anext(gen)
    assert item is not None
    # 스트림 중간에 aclose() 호출 시 RuntimeError 없이 안전하게 닫혀야 함
    await gen.aclose()
