import asyncio
import pytest
import sys
import os
import logging

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import RAGSystem
from core.session.manager import SessionManager

# Setup basic logging to see the output
logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_engine_recompilation_on_loop_change():
    """Verify that RAGSystem re-compiles the graph when event loop changes."""
    session_id = "test_loop_change"
    SessionManager.init_session(session_id)
    # Set dummy file_hash to allow engine building
    SessionManager.set("file_hash", "dummy_hash", session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    # Initialize engine in current loop
    engine1 = await rag._get_rag_engine()
    assert engine1 is not None, "Engine 1 should not be None"
    loop1 = asyncio.get_running_loop()
    loop1_id = id(loop1)

    # Simulate a new loop (e.g., a different thread's loop)
    def thread_worker(results):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:

            async def get_engine():
                # Invalidate the singleton graph cache so build_graph() recompiles
                from core.graph_builder import invalidate_graph_cache

                invalidate_graph_cache()
                # This call should trigger re-compilation because loop id will be different
                engine2 = await rag._get_rag_engine()
                return engine2, id(asyncio.get_running_loop())

            results["engine2"], results["loop2_id"] = new_loop.run_until_complete(
                get_engine()
            )
        finally:
            new_loop.close()

    import threading

    results = {}
    thread = threading.Thread(target=thread_worker, args=(results,))
    thread.start()
    thread.join()

    loop2_id = results.get("loop2_id")
    engine2 = results.get("engine2")

    assert loop1_id != loop2_id
    assert engine1 is not engine2
