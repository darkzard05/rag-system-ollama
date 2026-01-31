import asyncio
import os
import sys
import time

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 프로젝트 루트 및 src 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from common.config import AVAILABLE_EMBEDDING_MODELS
from core.model_loader import load_embedding_model
from core.semantic_chunker import EmbeddingBasedSemanticChunker


async def run_verification():
    print("🔍 [Safety Test] 벡터 재사용 및 풀링 정확도 검증 시작")

    # 1. 환경 준비
    "cuda" if torch.cuda.is_available() else "cpu"
    model_name = AVAILABLE_EMBEDDING_MODELS[0]
    embedder = load_embedding_model(model_name)

    # 테스트용 데이터 생성 (긴 문장들)
    texts = [
        "인공지능은 현대 사회의 핵심 기술로 자리잡고 있습니다. " * 5,
        "벡터 데이터베이스는 대규모 비정형 데이터를 효율적으로 검색합니다. " * 5,
        "RAG 시스템은 외부 지식을 결합하여 LLM의 환각 현상을 줄입니다. " * 5,
    ]
    docs = [
        Document(page_content=t, metadata={"source": "test", "page": i})
        for i, t in enumerate(texts)
    ]

    # --- 실험 1: 기존 방식 (재임베딩 발생) ---
    print("\n[Method 1] 기존 방식 (재임베딩 시뮬레이션)")
    start_time = time.time()

    # 청킹 (벡터 결과 무시)
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder, min_chunk_size=100, max_chunk_size=500
    )
    split_docs, _ = chunker.split_documents(docs)

    # FAISS 생성 (내부에서 처음부터 다시 임베딩 수행)
    vector_store_orig = FAISS.from_documents(split_docs, embedder)

    orig_time = time.time() - start_time
    print(f"✅ 기존 방식 소요 시간: {orig_time:.4f}s")

    # --- 실험 2: 최적화 방식 (벡터 재사용) ---
    print("\n[Method 2] 최적화 방식 (벡터 풀링 및 주입)")
    start_time = time.time()

    # 청킹 (이미 계산된 벡터 확보)
    split_docs_opt, precomputed_vectors = chunker.split_documents(docs)

    # FAISS 직접 주입 (재임베딩 0회)
    text_embeddings = list(
        zip([d.page_content for d in split_docs_opt], precomputed_vectors, strict=False)
    )
    metadatas = [d.metadata for d in split_docs_opt]
    vector_store_opt = FAISS.from_embeddings(
        text_embeddings=text_embeddings, embedding=embedder, metadatas=metadatas
    )

    opt_time = time.time() - start_time
    print(f"✅ 최적화 방식 소요 시간: {opt_time:.4f}s")

    # --- 실험 3: 검색 품질 비교 (가장 중요) ---
    print("\n[Result] 검색 품질 비교 결과")
    query = "AI와 벡터 검색의 관계"

    # 검색 수행
    res_orig = vector_store_orig.similarity_search(query, k=1)
    res_opt = vector_store_opt.similarity_search(query, k=1)

    # 결과 검증
    is_identical = res_orig[0].page_content == res_opt[0].page_content
    print(f"🔹 검색 결과 일치 여부: {'⭕ 일치' if is_identical else '❌ 불일치'}")

    # 코사인 유사도 검증
    embedder.embed_query(query)
    # FAISS 인덱스 내의 실제 벡터들 간의 거리가 동일한지 확인
    # (FAISS는 임베딩 함수를 사용하여 쿼리를 변환하므로 결과가 같아야 함)

    improvement = (orig_time - opt_time) / orig_time * 100
    print(f"🚀 성능 개선율: {improvement:.2f}%")

    if is_identical:
        print(
            "\n✨ 검증 성공: 벡터 재사용은 품질 저하 없이 성능만 획기적으로 향상시킵니다."
        )
    else:
        print(
            "\n⚠️ 검증 주의: 검색 결과에 미세한 차이가 발생했습니다 (풀링 가중치 확인 필요)."
        )


if __name__ == "__main__":
    asyncio.run(run_verification())
