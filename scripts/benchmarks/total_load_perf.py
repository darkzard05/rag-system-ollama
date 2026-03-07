
import asyncio
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager

async def measure_load_time():
    pdf_path = "tests/data/2201.07520v1.pdf"
    file_name = "2201.07520v1.pdf"
    
    print(f"[*] Testing RAGSystem.load_document with current config...")
    rag = RAGSystem(session_id="perf_test")
    embedder = await ModelManager.get_embedder()
    
    start_time = time.perf_counter()
    await rag.load_document(pdf_path, file_name, embedder)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    print(f"[*] Total load_document time: {total_time:.2f} seconds")
    
    # 만약 청크 개수를 알 수 있다면 출력
    if hasattr(rag, 'vector_store') and rag.vector_store:
        # FAISS index size
        try:
            count = rag.vector_store.index.ntotal
            print(f"[*] Total chunks indexed: {count}")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(measure_load_time())
