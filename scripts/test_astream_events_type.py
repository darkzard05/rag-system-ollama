import asyncio
from core.graph_builder import build_graph
from core.session import SessionManager


async def test_astream_events():
    SessionManager.init_session(session_id="test_session")
    rag_engine = await build_graph()

    query = "안녕"
    chat_history = []
    config = {"configurable": {"thread_id": "test_session"}}

    print("Starting astream_events...")
    async for event in rag_engine.astream_events(
        {"input": query, "chat_history": chat_history},
        config=config,
        version="v2",
    ):
        print(f"Event type: {type(event)}")
        print(f"Event: {event}")
        break


if __name__ == "__main__":
    asyncio.run(test_astream_events())
