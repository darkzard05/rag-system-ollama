import os
import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL

async def run_full_pipeline_test():
    print("[E2E] RAG Pipeline Integration Test Started")
    
    session_id = "test-session-final"
    rag = RAGSystem(session_id=session_id)
    
    print("1. Loading Models...")
    try:
        embedder = load_embedding_model()
        llm = load_llm(DEFAULT_OLLAMA_MODEL)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print(f"2. Indexing Document: {os.path.basename(test_pdf)}")
    
    start_time = asyncio.get_event_loop().time()
    msg, cache_used = await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)
    load_time = asyncio.get_event_loop().time() - start_time
    print(f"   Result: {msg} (Cache: {cache_used}) | Time: {load_time:.2f}s")

    test_query = "What is the core idea of CM3 model?"
    print(f"\n3. Querying: '{test_query}'")
    
    start_time = asyncio.get_event_loop().time()
    result = await rag.aquery(test_query, llm=llm)
    query_time = asyncio.get_event_loop().time() - start_time
    
    print(f"\n4. Test Results (Time: {query_time:.2f}s)")
    print("-" * 50)
    print(f"Intent: {result.get('route_decision', 'N/A')}")
    print(f"Documents: {len(result.get('documents', []))} blocks")
    print("\n[Answer]")
    print(result.get("response", "Failed to generate answer"))
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_full_pipeline_test())