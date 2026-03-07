
import asyncio
import time
import sys
import os
import torch
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.document_processor import load_pdf_docs
from core.chunking import split_documents
from core.model_loader import ModelManager
from core.session import SessionManager
from common.config import SEMANTIC_CHUNKER_CONFIG, TEXT_SPLITTER_CONFIG

async def test_performance():
    print("="*80)
    print("RAG Embedding Bottleneck Analysis")
    print("="*80)
    
    # 0. 세션 및 환경 설정
    SessionManager.init_session("bench_session")
    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = "2201.07520v1.pdf"
    
    # 1. 문서 로드 (PyMuPDF4LLM)
    print(f"[*] Loading PDF: {file_name}")
    docs = load_pdf_docs(pdf_path, file_name)
    print(f"    - Pages: {len(docs)}")
    
    # 2. 임베딩 모델 준비 (Local E5)
    model_name = "intfloat/multilingual-e5-small"
    print(f"[*] Loading Embedder: {model_name}")
    embedder = await ModelManager.get_embedder(model_name)
    
    # --- Case 1: Standard Recursive Chunking ---
    print("\n[Case 1] Standard Recursive Chunking + Embedding")
    SEMANTIC_CHUNKER_CONFIG["enabled"] = False
    
    start_t = time.time()
    split_docs, vectors = await split_documents(docs, embedder, session_id="bench_session")
    dur = time.time() - start_t
    
    print(f"    - Chunks: {len(split_docs)}")
    print(f"    - Total Time: {dur:.2f}s")
    if vectors is not None:
        print(f"    - Vectors: {len(vectors)}")

    # --- Case 2: Semantic Chunking (No Cache) ---
    print("\n[Case 2] Semantic Chunking (Fresh, No Cache)")
    SEMANTIC_CHUNKER_CONFIG["enabled"] = True
    
    # VRAM/Cache 클리어
    await ModelManager.clear_vram()
    embedder = await ModelManager.get_embedder(model_name)
    
    start_t = time.time()
    split_docs_s, vectors_s = await split_documents(docs, embedder, session_id="bench_session")
    dur_s = time.time() - start_t
    
    print(f"    - Chunks: {len(split_docs_s)}")
    print(f"    - Total Time: {dur_s:.2f}s")
    print(f"    - Slowdown Factor: {dur_s / dur:.1f}x slower than standard")

    # --- Case 3: Semantic Chunking (Cached) ---
    print("\n[Case 3] Semantic Chunking (With Cache)")
    # 캐시된 상태로 재실행
    start_t = time.time()
    split_docs_c, vectors_c = await split_documents(docs, embedder, session_id="bench_session")
    dur_c = time.time() - start_t
    
    print(f"    - Total Time: {dur_c:.2f}s")
    print(f"    - Cache Speedup: {dur_s / dur_c:.1f}x faster with cache")

    print("\n" + "="*80)
    print("Analysis Summary:")
    print(f"1. Standard: {dur:.2f}s")
    print(f"2. Semantic (New): {dur_s:.2f}s")
    print(f"3. Semantic (Cached): {dur_c:.2f}s")
    print("="*80)

if __name__ == "__main__":
    # 윈도우 비동기 루프 이슈 방지
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_performance())
