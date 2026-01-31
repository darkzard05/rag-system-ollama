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

from common.config import SEMANTIC_CHUNKER_CONFIG
from core.rag_core import _load_pdf_docs, _split_documents


async def run_comparison():
    print("🚀 [Benchmark] Semantic Chunking vs Basic Chunking 비교 테스트")

    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ 테스트용 PDF 파일이 없습니다: {pdf_path}")
        return

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )

    # 원본 문서 로드 (공통)
    docs = _load_pdf_docs(pdf_path, "benchmark.pdf")

    results = {}

    # --- Case A: Basic Chunking ---
    print("\n[Case A] Basic Chunking (RecursiveCharacterTextSplitter) 실행 중...")
    # 임시로 설정 변경
    SEMANTIC_CHUNKER_CONFIG["enabled"] = False

    start_time = time.time()
    split_docs_a, _ = _split_documents(docs, embedder)
    # Basic은 벡터를 반환하지 않으므로 여기서 임베딩 수행
    texts_a = [d.page_content for d in split_docs_a]
    vectors_a = embedder.embed_documents(texts_a)
    _ = FAISS.from_embeddings(
        zip(texts_a, vectors_a, strict=False),
        embedder,
        metadatas=[d.metadata for d in split_docs_a],
    )
    time_a = time.time() - start_time

    results["Basic"] = {
        "time": time_a,
        "chunk_count": len(split_docs_a),
        "avg_len": sum(len(d.page_content) for d in split_docs_a) / len(split_docs_a),
    }

    # --- Case B: Semantic Chunking ---
    print("\n[Case B] Semantic Chunking (Vector Reuse) 실행 중...")
    # 임시로 설정 변경
    SEMANTIC_CHUNKER_CONFIG["enabled"] = True

    start_time = time.time()
    # 의미론적 청킹은 분할 과정에서 벡터를 이미 계산함
    split_docs_b, vectors_b = _split_documents(docs, embedder)

    # 벡터 재사용하여 FAISS 생성 (추가 임베딩 호출 없음)
    if vectors_b:
        vectors_np = [np.array(v) for v in vectors_b]
        _ = FAISS.from_embeddings(
            zip([d.page_content for d in split_docs_b], vectors_np, strict=False),
            embedder,
            metadatas=[d.metadata for d in split_docs_b],
        )
    time_b = time.time() - start_time

    results["Semantic"] = {
        "time": time_b,
        "chunk_count": len(split_docs_b),
        "avg_len": sum(len(d.page_content) for d in split_docs_b) / len(split_docs_b),
    }

    # --- 결과 출력 ---
    print("\n" + "=" * 60)
    print(f"{'지표':<20} | {'Basic (규칙 기반)':<18} | {'Semantic (의미론적)':<18}")
    print("-" * 60)
    print(
        f"{'소요 시간(초)':<20} | {results['Basic']['time']:<18.2f} | {results['Semantic']['time']:<18.2f}"
    )
    print(
        f"{'생성된 청크 수':<20} | {results['Basic']['chunk_count']:<18} | {results['Semantic']['chunk_count']:<18}"
    )
    print(
        f"{'평균 청크 길이':<20} | {results['Basic']['avg_len']:<18.1f} | {results['Semantic']['avg_len']:<18.1f}"
    )
    print("=" * 60)

    # 분석 의견
    print("\n📊 분석 결과:")
    if results["Semantic"]["time"] > results["Basic"]["time"]:
        overhead = (
            (results["Semantic"]["time"] - results["Basic"]["time"])
            / results["Basic"]["time"]
            * 100
        )
        print(f"1. 시간 비용: 의미론적 청킹이 약 {overhead:.1f}% 더 소요됩니다.")
    else:
        print("1. 시간 비용: 벡터 재사용 덕분에 의미론적 청킹이 오히려 효율적입니다.")

    print(
        f"2. 구조적 차이: Basic은 고정 크기로 쪼개지만, Semantic은 {results['Semantic']['chunk_count']}개의 '의미 단위'로 묶었습니다."
    )
    print(
        "3. 효용성: 현재 구조(Vector Reuse)는 의미론적 청킹 시 발생하는 임베딩 비용을 FAISS 인덱싱 단계에서 100% 회수하므로 매우 효율적입니다."
    )


if __name__ == "__main__":
    asyncio.run(run_comparison())
