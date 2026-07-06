import asyncio
import sys
import os

# Add src directory to path


from src.common.schemas import PerformanceStats
from src.common.performance_adapter import adapt_llm_metadata
from src.common.streaming import StreamingResponseHandler, StreamChunk


async def test_adapter():
    print("Testing PerformanceAdapter...")
    # Case 1: Ollama metadata
    ollama_meta = {
        "prompt_eval_count": 100,
        "eval_count": 200,
        "total_duration": 2000000000,
    }
    stats = adapt_llm_metadata(ollama_meta)
    assert stats.input_token_count == 100
    assert stats.token_count == 200
    assert stats.total_time == 2.0
    print("✅ Ollama metadata adapter passed")

    # Case 2: Generic metadata
    generic_meta = {"input_token_count": 50, "token_count": 150}
    stats = adapt_llm_metadata(generic_meta)
    assert stats.input_token_count == 50
    assert stats.token_count == 150
    print("✅ Generic metadata adapter passed")


async def test_streaming_handler():
    print("\nTesting StreamingResponseHandler...")
    handler = StreamingResponseHandler()

    # Mock event stream
    async def mock_event_stream():
        yield "custom", {"status": "thinking..."}
        yield "messages", ("chunk1", {})
        yield "messages", ("chunk2", {})
        yield (
            "updates",
            {
                "generate": {
                    "performance": {"prompt_eval_count": 100, "eval_count": 200}
                }
            },
        )
        yield "messages", ("chunk3", {})

    # We need a dummy adaptive controller or None
    chunks = []
    async for chunk in handler.stream_graph_events(mock_event_stream()):
        chunks.append(chunk)

    # Find final chunk
    final_chunk = next((c for c in chunks if c.is_final), None)
    assert final_chunk is not None
    assert final_chunk.performance is not None

    perf = final_chunk.performance
    print(f"Final Performance: {perf}")

    assert perf["input_token_count"] == 100
    assert perf["token_count"] == 200
    assert "tps" in perf
    assert perf["tps"] > 0
    print("✅ StreamingResponseHandler performance collection passed")


if __name__ == "__main__":
    asyncio.run(test_adapter())
    asyncio.run(test_streaming_handler())
