import asyncio
import sys
import os

# Add src directory to PYTHONPATH
sys.path.append(os.path.abspath("src"))

from src.core.rag_core import RAGSystem
from src.core.graph_builder import get_rag_graph


async def verify_identity():
    print("--- Identity Verification ---")
    rag1 = RAGSystem(session_id="session_1")
    rag2 = RAGSystem(session_id="session_2")

    engine1 = await rag1._get_rag_engine()
    engine2 = await rag2._get_rag_engine()

    print(f"Engine 1 ID: {id(engine1)}")
    print(f"Engine 2 ID: {id(engine2)}")

    if engine1 is engine2:
        print("✅ SUCCESS: Both sessions share the same singleton engine instance.")
    else:
        print("❌ FAILURE: Sessions have different engine instances.")


if __name__ == "__main__":
    asyncio.run(verify_identity())
