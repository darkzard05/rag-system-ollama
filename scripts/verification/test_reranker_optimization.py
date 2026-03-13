import asyncio
import time
import os
import sys

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))

from langchain_core.runnables import RunnableConfig
from langchain_core.embeddings import Embeddings
from core.graph_builder import retrieve_and_rerank
from core.document_processor import load_pdf_docs
from core.retriever_factory import create_vector_store, create_bm25_retriever
from core.model_loader import ModelManager

async def verify_reranker_optimization():
    # 1. 테스트 환경 준비
    pdf_path = "tests/data/2201.07520v1.pdf"
    query = "CM3 모델의 인프라 및 트레이닝 데이터 셋 구성은 어떻게 되어있나요?"
    
    print(f"\n--- [RERANKER TEST] Loading PDF: {pdf_path} ---")
    docs = load_pdf_docs(pdf_path, "test.pdf")
    
    # 2. 임베딩 모델 및 리트리버 로드
    embedder = await ModelManager.get_embedder()
    # 청킹 (간소화)
    from core.chunking import split_documents
    doc_splits, vectors = await split_documents(docs, embedder)
    
    vector_store = create_vector_store(doc_splits, embedder, vectors=vectors)
    bm25_retriever = create_bm25_retriever(doc_splits)
    
    # 3. 리랭킹 노드 실행 및 시간 측정
    print(f"\n--- [RERANKER TEST] Query: '{query}' ---")
    
    config = {
        "configurable": {
            "bm25_retriever": bm25_retriever,
            "faiss_retriever": vector_store.as_retriever(search_kwargs={"k": 10}),
        }
    }
    
    state = {"input": query, "retry_count": 0, "search_queries": []}
    writer = lambda x: print(f"  [STATUS] {x.get('status', '')}") if x.get('status') else None

    # 시작 시간 측정
    start_time = time.time()
    
    result = await retrieve_and_rerank(state, config, writer)
    
    duration = time.time() - start_time
    print(f"\n--- [RESULT] Total Retrieve & Rerank Duration: {duration:.2f}s ---")
    
    relevant_docs = result.get("relevant_docs", [])
    print(f"Top {len(relevant_docs)} Docs Selected:")
    
    for i, doc in enumerate(relevant_docs[:3]):
        page = doc.metadata.get("page", "?")
        content_preview = doc.page_content[:150].replace("\n", " ")
        # 병합 확인: 청크 인덱스 범위 확인
        chunk_indices = doc.metadata.get("chunk_index", "N/A")
        print(f"  {i+1}. [Page {page}] {content_preview}...")
        
    # 4. 검증 포인트 확인
    if len(relevant_docs) > 0:
        print("\n[VERIFICATION SUCCESS]")
        print("- 리랭킹 노드가 성공적으로 실행됨.")
        print(f"- 최종 선별된 문서 수: {len(relevant_docs)}")
    else:
        print("\n[VERIFICATION FAILED] 선별된 문서가 없습니다.")

if __name__ == "__main__":
    asyncio.run(verify_reranker_optimization())
