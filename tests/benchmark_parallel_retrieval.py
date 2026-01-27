
import asyncio
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

async def benchmark_parallel_retrieval():
    print("🚀 [Benchmark] 병렬 하이브리드 검색 성능 테스트")
    
    # 1. 데이터 준비 (500개 청크)
    print("📊 테스트 데이터 준비 중...")
    docs = [
        Document(page_content=f"테스트 문장 {i}입니다. 검색 성능 측정을 위한 데이터입니다.", metadata={"id": i})
        for i in range(500)
    ]
    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_vs = FAISS.from_documents(docs, embedder)
    faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    query = "테스트 문장 100번에 대해 알려줘"

    # --- [Test 1] 순차 검색 (Sequential) ---
    print("\n--- [Sequential] 순차 검색 실행 ---")
    start_time = time.time()
    
    res1 = bm25_retriever.invoke(query)
    res2 = faiss_retriever.invoke(query)
    
    seq_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {seq_time:.4f}초")

    # --- [Test 2] 병렬 검색 (Parallel) ---
    print("\n--- [Parallel] 병렬 검색 실행 ---")
    start_time = time.time()
    
    # asyncio.gather를 사용하여 동시 실행 (ainvoke 사용)
    # BM25Retriever는 보통 동기이므로 asyncio.to_thread 활용 시뮬레이션
    results = await asyncio.gather(
        asyncio.to_thread(bm25_retriever.invoke, query),
        faiss_retriever.ainvoke(query)
    )
    
    par_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {par_time:.4f}초")

    # 2. 결과 분석
    improvement = ((seq_time - par_time) / seq_time) * 100
    print("\n" + "="*40)
    print(f"📈 성능 개선 결과")
    print(f"  - 순차 방식: {seq_time:.4f}초")
    print(f"  - 병렬 방식: {par_time:.4f}초")
    print(f"  - 개선율: {improvement:.1f}%")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(benchmark_parallel_retrieval())
