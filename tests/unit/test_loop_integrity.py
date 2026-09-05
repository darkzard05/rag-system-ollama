import asyncio
import logging
import os
import sys
import threading

import pytest

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import RAGSystem
from core.session import SessionManager

# Setup basic logging to see the output
logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_engine_reused_across_loop_change():
    """Verify that RAGSystem reuses the same engine across event-loop changes.

    Engine validity is keyed solely on the document file_hash (see commit
    74957c6): LangGraph compiled graphs are not loop-bound, so reusing the
    cached engine across loops is the intended behavior. invalidate_graph_cache()
    only busts the graph-build cache and must NOT force engine recompilation
    while the file_hash is unchanged.
    """
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

    # Run on a different loop (e.g., a different thread's loop)
    def thread_worker(results):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:

            async def get_engine():
                # Invalidate the singleton graph cache ONLY — should NOT
                # invalidate the engine cache because file_hash is unchanged.
                from core.graph.graph_builder import invalidate_graph_cache

                invalidate_graph_cache()
                engine2 = await rag._get_rag_engine()
                return engine2, id(asyncio.get_running_loop())

            results["engine2"], results["loop2_id"] = new_loop.run_until_complete(
                get_engine()
            )
        finally:
            new_loop.close()

    results = {}
    thread = threading.Thread(target=thread_worker, args=(results,))
    thread.start()
    thread.join()

    loop2_id = results.get("loop2_id")
    engine2 = results.get("engine2")

    # The two calls ran on different event loops...
    assert loop1_id != loop2_id
    # ...but the engine is reused because file_hash is unchanged (cross-loop reuse).
    assert engine2 is not None
    assert engine1 is engine2
