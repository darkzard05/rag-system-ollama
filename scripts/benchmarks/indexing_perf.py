
import asyncio
import numpy as np
import time
import os
import sys
import faiss
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.retriever_factory import create_vector_store
from services.optimization.index_optimizer import get_index_optimizer

class MockEmbedder(Embeddings):
    def __init__(self, dimension=768):
        self.dimension = dimension
    def embed_documents(self, texts):
        return [np.random.random(self.dimension).astype('float32').tolist() for _ in texts]
    def embed_query(self, text):
        return np.random.random(self.dimension).astype('float32').tolist()

def measure_index_memory(vector_store):
    temp_file = "vstore_bench.faiss"
    faiss.write_index(vector_store.index, temp_file)
    size = os.path.getsize(temp_file)
    os.remove(temp_file)
    return size

async def run_indexing_benchmark(n_docs: int = 5000):
    dimension = 768
    embedder = MockEmbedder(dimension)
    docs = [Document(page_content="Bench content " + str(i), metadata={"page": i}) for i in range(n_docs)]
    
    print("--- [Benchmark] 인덱싱 성능 측정 시작 (" + str(n_docs) + "개) ---")
    
    # 1. 수동 최적화 (Pruning) 시간 측정
    optimizer = get_index_optimizer() 
    vectors = np.array(embedder.embed_documents([d.page_content for d in docs])).astype('float32')
    
    start = time.time()
    doc_splits, opt_vectors, _, stats = optimizer.optimize_index(docs, list(vectors))
    opt_dur = time.time() - start
    
    # 2. FAISS 인덱스 구축 시간 및 메모리 측정
    start = time.time()
    vstore = create_vector_store(doc_splits, embedder, vectors=opt_vectors)
    vstore_dur = time.time() - start
    
    mem_mb = measure_index_memory(vstore) / (1024 * 1024)
    raw_size_mb = (n_docs * dimension * 4) / (1024 * 1024)
    
    print("
--- 성능 결과 ---")
    print("중복 제거 시간: " + str(round(opt_dur, 4)) + "s")
    print("인덱스 구축 시간: " + str(round(vstore_dur, 4)) + "s")
    print("최종 인덱스 크기: " + str(round(mem_mb, 2)) + " MB (원본 대비 " + str(round(mem_mb/raw_size_mb*100, 1)) + "%)")
    
    if mem_mb < raw_size_mb * 0.4:
        print("상태: SQ8 양자화 정상 작동 중")
    else:
        print("상태: 양자화 미적용 혹은 오버헤드 높음")

if __name__ == "__main__":
    asyncio.run(run_indexing_benchmark(5000))
