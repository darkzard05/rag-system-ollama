import os
import sys
import time
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.config import OLLAMA_BASE_URL
from core.document_processor import load_pdf_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def run_final_benchmark():
    print("="*95)
    print(f"Final Embedding Battle: Nomic vs BGE-M3 (via Ollama) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*95)

    # 문서 로드
    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    try:
        raw_docs = load_pdf_docs(pdf_path, "2201.07520v1.pdf")
    except Exception as e:
        print(f"PDF Load Error: {e}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(raw_docs)
    texts = [d.page_content for d in split_docs]
    print(f"Total Chunks: {len(texts)}")

    models = ["nomic-embed-text:latest", "bge-m3:latest"]
    results = []

    for m_id in models:
        print(f"\n>>> Testing: {m_id}")
        from langchain_ollama import OllamaEmbeddings
        try:
            embedder = OllamaEmbeddings(model=m_id, base_url=OLLAMA_BASE_URL)
            # 웜업
            embedder.embed_query("warmup")

            start_t = time.time()
            vectors = embedder.embed_documents(texts)
            vector_time = time.time() - start_t
            
            dim = len(vectors[0])
            avg_ms = (vector_time / len(texts)) * 1000
            
            print(f"   - Total Time: {vector_time:.2f}s")
            print(f"   - Avg Latency: {avg_ms:.2f}ms")
            print(f"   - Dimension: {dim}")
            
            results.append({
                "name": m_id,
                "total_time": vector_time,
                "avg_ms": avg_ms,
                "dim": dim
            })
        except Exception as e:
            print(f"   - Testing failed for {m_id}: {e}")

    print("\n" + "="*80)
    print(f"{'Model Name':<30} | {'Total(s)':<10} | {'Avg(ms)':<10} | {'Dim':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} | {r['total_time']:<10.2f} | {r['avg_ms']:<10.2f} | {r['dim']:<6}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_final_benchmark())
