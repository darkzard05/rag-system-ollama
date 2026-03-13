import asyncio
import time
import os
import sys
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.document_processor import load_pdf_docs
from core.chunking import split_documents
from core.retriever_factory import create_vector_store

class MockEmbedder(Embeddings):
    def __init__(self, dimension=768):
        self.dimension = dimension
    def embed_documents(self, texts):
        # 텍스트 길이에 따라 약간의 차이를 주어 중복 제거 테스트 가능하게 함
        return [np.random.random(self.dimension).astype('float32').tolist() for _ in texts]
    def embed_query(self, text):
        return np.random.random(self.dimension).astype('float32').tolist()

async def run_total_benchmark(file_path: str, label: str):
    print(f"\n[Benchmark] {label} 시작: {file_path}")
    embedder = MockEmbedder()
    
    # 1. 문서 로딩 및 파싱
    start_parse = time.perf_counter()
    docs = load_pdf_docs(file_path, Path(file_path).name)
    parse_time = time.perf_counter() - start_parse
    
    # 2. 청킹 및 벡터화
    start_chunk = time.perf_counter()
    doc_splits, vectors = await split_documents(docs, embedder)
    chunk_time = time.perf_counter() - start_chunk
    
    # 3. 인덱싱 (벡터 저장소 구축)
    start_index = time.perf_counter()
    vstore = create_vector_store(doc_splits, embedder, vectors=vectors)
    index_time = time.perf_counter() - start_index
    
    total_time = parse_time + chunk_time + index_time
    
    print(f"--- {label} 결과 ---")
    print(f"파싱 시간: {parse_time:.4f}s")
    print(f"청킹 시간: {chunk_time:.4f}s")
    print(f"인덱싱 시간: {index_time:.4f}s")
    print(f"총 소요 시간: {total_time:.4f}s")
    print(f"생성된 청크 수: {len(doc_splits)}")
    
    return {
        "parse_time": parse_time,
        "chunk_time": chunk_time,
        "index_time": index_time,
        "total_time": total_time,
        "chunks": len(doc_splits)
    }

if __name__ == "__main__":
    test_file = "tests/data/2201.07520v1.pdf"
    if not Path(test_file).exists():
        print(f"Error: {test_file} not found.")
        sys.exit(1)
        
    results = asyncio.run(run_total_benchmark(test_file, "BASELINE"))
    # 결과를 파일로 저장하여 나중에 비교
    import json
    with open("logs/baseline_perf.json", "w") as f:
        json.dump(results, f)
