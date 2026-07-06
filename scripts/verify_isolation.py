import asyncio
import sys
import os

sys.path.append(os.path.abspath("src"))

from src.core.rag_core import RAGSystem
from src.core.graph_builder import get_rag_graph


async def verify_isolation():
    print("--- Isolation & Persistence Verification ---")

    # Simulation of two different users/sessions
    rag1 = RAGSystem(session_id="user_alpha")
    rag2 = RAGSystem(session_id="user_beta")

    # We will mock the actual ainvoke to see what config is passed
    # because we might not have a live LLM for a simple unit test.
    # But we want to see if the singleton engine is used with different configs.

    engine = get_rag_graph()

    # Simulate config preparation
    # In real aquery, _prepare_config is called.
    config1 = {"configurable": {"thread_id": "user_alpha", "session_id": "user_alpha"}}
    config2 = {"configurable": {"thread_id": "user_beta", "session_id": "user_beta"}}

    print(f"Session Alpha Config: {config1}")
    print(f"Session Beta Config: {config2}")

    # Since we are using a singleton engine, we just need to ensure
    # that LangGraph handles different thread_ids correctly.
    # We can test this by checking if the checkpointer stores different states.

    from langgraph.checkpoint.memory import InMemorySaver
    # We can't easily access the private _GLOBAL_CHECKPOINTER without importing it
    # but we can trust LangGraph's InMemorySaver if the thread_id is different.

    # To truly verify persistence, we can use a dummy state.
    try:
        # Use the actual engine to update state for two different threads
        engine.update_state(config1, {"input": "Hello from Alpha"})
        engine.update_state(config2, {"input": "Hello from Beta"})

        state1 = engine.get_state(config1)
        state2 = engine.get_state(config2)

        print(f"State 1: {state1.values.get('input')}")
        print(f"State 2: {state2.values.get('input')}")

        if (
            state1.values.get("input") == "Hello from Alpha"
            and state2.values.get("input") == "Hello from Beta"
        ):
            print(
                "✅ SUCCESS: Session states are isolated and persisted correctly via thread_id."
            )
        else:
            print("❌ FAILURE: Session state leak detected.")

    except Exception as e:
        print(f"❌ ERROR during isolation test: {e}")


if __name__ == "__main__":
    asyncio.run(verify_isolation())
