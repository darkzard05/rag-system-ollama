import asyncio
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 기존 로직 임포트 (수정된 버전)
from core.rag_core import _load_pdf_docs, _split_documents


async def benchmark_indexing():
    print("🚀 [Benchmark] 임베딩 재사용 성능 테스트 시작")

    # 1. 환경 준비
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ 테스트용 PDF 파일이 없습니다: {pdf_path}")
        return

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 사용 디바이스: {device}")

    embedder = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )

    # 2. 문서 로드 및 청킹
    print("📄 문서 로드 및 청킹 중...")
    docs = _load_pdf_docs(pdf_path, "benchmark.pdf")
    doc_splits = _split_documents(docs, embedder)
    texts = [d.page_content for d in doc_splits]
    print(f"📊 총 청크 수: {len(doc_splits)}")

    # 3. 최적화 전 방식 (임베딩 2회) 측정
    print("\n--- [Test 1] 이전 방식 (이중 임베딩) 시뮬레이션 ---")
    start_time = time.time()

    # 1회차: 최적화 로직용 임베딩
    print("  Step 1: 최적화용 임베딩 생성 중...")
    embedder.embed_documents(texts)

    # 2회차: FAISS 생성 (내부에서 다시 임베딩 수행)
    print("  Step 2: FAISS 생성 (강제 재임베딩) 중...")
    _ = FAISS.from_documents(doc_splits, embedder)

    old_method_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {old_method_time:.2f}초")

    # 4. 최적화 후 방식 (임베딩 1회 + 재사용) 측정
    print("\n--- [Test 2] 현재 방식 (임베딩 재사용) 측정 ---")
    start_time = time.time()

    # 1회차: 임베딩 생성
    print("  Step 1: 임베딩 생성 중...")
    vectors_2 = embedder.embed_documents(texts)
    vectors_np = [np.array(v) for v in vectors_2]

    # 2회차: 벡터 재사용하여 FAISS 생성
    print("  Step 2: 벡터 재사용하여 FAISS 생성 중...")
    text_embeddings = zip(
        [d.page_content for d in doc_splits], vectors_np, strict=False
    )
    _ = FAISS.from_embeddings(
        text_embeddings, embedder, metadatas=[d.metadata for d in doc_splits]
    )

    new_method_time = time.time() - start_time
    print(f"⏱️ 소요 시간: {new_method_time:.2f}초")

    # 5. 결과 비교
    improvement = ((old_method_time - new_method_time) / old_method_time) * 100
    print("\n" + "=" * 40)
    print("📈 성능 개선 결과")
    print(f"  - 이전 방식: {old_method_time:.2f}초")
    print(f"  - 현재 방식: {new_method_time:.2f}초")
    print(f"  - 절감 시간: {old_method_time - new_method_time:.2f}초")
    print(f"  - 개선율: {improvement:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(benchmark_indexing())
