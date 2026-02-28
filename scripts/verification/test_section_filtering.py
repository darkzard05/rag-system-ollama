import os
import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
from common.logging_config import setup_logging

async def test_section_filtering():
    setup_logging(log_level="INFO")
    
    print("\n" + "="*60)
    print("🧪 Paper Section Filtering (TOC/References) Verification")
    print("="*60)

    session_id = "filter-test-session"
    rag = RAGSystem(session_id=session_id)
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)

    print(f"\n[STEP 1] Indexing paper: {file_name}")
    msg, _ = await rag.build_pipeline(test_pdf, file_name, embedder)
    print(f"✅ Indexing Result: {msg}")

    from core.resource_pool import get_resource_pool
    from core.session import SessionManager
    
    file_hash = SessionManager.get("file_hash", session_id=session_id)
    vector_store, _ = await get_resource_pool().get(file_hash)
    
    total_chunks = vector_store.index.ntotal if vector_store else 0
    print(f"\n[STEP 2] Verifying reduction in chunks:")
    print(f"📊 Current total chunks in index: {total_chunks}")
    print(" (Note: Original was 326 chunks. Less means filtering worked.)")

    print("\n[STEP 3] Verifying search quality (No References):")
    query = "List the papers cited in the references section about Attention"
    result = await rag.aquery(query, model_name=DEFAULT_OLLAMA_MODEL)
    
    docs = result.get("documents", [])
    has_ref_list = False
    for i, doc in enumerate(docs):
        content = doc.page_content.lower()
        if "et al." in content and content.count("\n") > 5:
            print(f"⚠️ Potential Reference list in Chunk {i+1}!")
            has_ref_list = True

    if not has_ref_list:
        print("✅ SUCCESS: No reference lists detected.")
    else:
        print("❌ FAILURE: Reference lists are still being indexed.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_section_filtering())
