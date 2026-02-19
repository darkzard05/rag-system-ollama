import os
import sys
import time
import asyncio
import torch
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_embedding_model, ModelManager
from common.config import OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from core.session import SessionManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 테스트 데이터 준비 (메모리 로드)
def get_test_documents():
    from core.document_processor import load_pdf_docs
    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    return load_pdf_docs(pdf_path, "2201.07520v1.pdf")

MODELS = [
    {"id": "intfloat/multilingual-e5-small", "name": "E5-Small (Local)", "backend": "onnx"},
    {"id": "BAAI/bge-m3", "name": "BGE-M3 (Local)", "backend": "default"}, # ONNX 대신 기본 백엔드 사용
    {"id": "nomic-embed-text:latest", "name": "Nomic (Ollama)", "backend": "ollama"}
]

def get_memory_info():
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / (1024 * 1024)
    vram = 0
    if torch.cuda.is_available():
        try:
            vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
        except: pass
    return ram, vram

async def run_benchmark():
    print("="*95)
    print(f"Embedding Architecture Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*95)

    # 0. 공통 문서 로드 및 분할 (비교의 공정성을 위해 동일 청크 사용)
    print("Pre-processing documents...")
    raw_docs = get_test_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(raw_docs)
    texts = [d.page_content for d in split_docs]
    print(f"Total Chunks: {len(texts)}")

    results = []

    for model_info in MODELS:
        m_id = model_info["id"]
        m_name = model_info["name"]
        m_backend = model_info["backend"]

        print(f"\n>>> Testing Model: {m_name}")
        ModelManager.clear_vram()
        time.sleep(1)
        
        # 1. 모델 로딩 (In-process vs Ollama)
        start_t = time.time()
        try:
            if m_backend == "ollama":
                from langchain_ollama import OllamaEmbeddings
                embedder = OllamaEmbeddings(model=m_id, base_url=OLLAMA_BASE_URL)
            else:
                from langchain_huggingface import HuggingFaceEmbeddings
                # BGE-M3의 경우 ONNX 이슈 회避를 위해 backend 강제 지정
                model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
                if m_backend == "onnx":
                    model_kwargs["backend"] = "onnx"
                
                embedder = HuggingFaceEmbeddings(
                    model_name=m_id,
                    model_kwargs=model_kwargs,
                    encode_kwargs={"batch_size": 32}
                )
            load_time = time.time() - start_t
            print(f"   - Load Time: {load_time:.2f}s")
        except Exception as e:
            print(f"   !!! Load Failed: {e}")
            continue

        # 웜업
        try: embedder.embed_query("warmup")
        except: pass

        ram_start, vram_start = get_memory_info()

        # 2. 순수 임베딩 시간 측정
        print(f"   - Vectorizing {len(texts)} chunks...")
        start_t = time.time()
        vectors = embedder.embed_documents(texts)
        vector_time = time.time() - start_t
        
        ram_end, vram_end = get_memory_info()

        # 3. 검색 시간 측정 (FAISS)
        import faiss
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        v_np = np.array(vectors).astype("float32")
        d = v_np.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(v_np)
        index.add(v_np)
        
        # 가상의 리트리버 시뮬레이션
        q_vec = np.array(embedder.embed_query("Search test")).astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)
        
        q_start = time.time()
        for _ in range(10): # 10회 평균
            index.search(q_vec, 5)
        search_time_ms = (time.time() - q_start) / 10 * 1000

        res = {
            "name": m_name,
            "vector_time": vector_time,
            "search_ms": search_time_ms,
            "ram_usage": ram_end - ram_start,
            "vram_usage": vram_end - vram_start,
            "dim": d
        }
        results.append(res)
        print(f"   - Done: Vectorize {vector_time:.2f}s, Search {search_time_ms:.4f}ms")

    # 최종 결과 출력
    print("\n" + "="*95)
    header = f"{'Model Name':<20} | {'Embed(s)':<10} | {'Search(ms)':<10} | {'Dim':<6} | {'RAM(MB)':<10} | {'VRAM(MB)':<10}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<20} | {r['vector_time']:<10.2f} | {r['search_ms']:<10.4f} | {r['dim']:<6} | {r['ram_usage']:<10.1f} | {r['vram_usage']:<10.1f}")
    print("="*95)

if __name__ == "__main__":
    asyncio.run(run_benchmark())

