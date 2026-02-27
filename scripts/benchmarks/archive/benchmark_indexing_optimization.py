import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from common.config import AVAILABLE_EMBEDDING_MODELS
from core.model_loader import load_embedding_model
from core.rag_core import _create_vector_store, _load_pdf_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")


async def run_benchmark():
    print("\n=== 인덱싱 최적화(임베딩 재사용) 성능 측정 ===")

    # 1. 준비
    embedder = load_embedding_model(AVAILABLE_EMBEDDING_MODELS[0])
    test_pdf = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(test_pdf):
        print("테스트 PDF가 없습니다.")
        return

    # 2. 문서 로드 및 분할 (의미론적 청킹 활성화 가정)
    docs = _load_pdf_docs(test_pdf, "test.pdf")

    # [A] 최적화 미적용 시뮬레이션 (순수 FAISS 생성 시간 측정)
    print("\n[A] 기존 방식: FAISS가 임베딩을 처음부터 다시 계산")
    start_a = time.time()
    _ = _create_vector_store(docs, embedder, vectors=None)
    duration_a = time.time() - start_a
    print(f">> 소요 시간: {duration_a:.4f}초")

    # [B] 최적화 적용 (이미 계산된 벡터 주입)
    print("\n[B] 최적화 방식: 이미 계산된 벡터를 FAISS에 주입")
    # 먼저 벡터 확보
    texts = [d.page_content for d in docs]
    precomputed_vectors = [np.array(v) for v in embedder.embed_documents(texts)]

    start_b = time.time()
    _ = _create_vector_store(docs, embedder, vectors=precomputed_vectors)
    duration_b = time.time() - start_b
    print(f">> 소요 시간: {duration_b:.4f}초")

    # 3. 결과 분석
    improvement = (1 - (duration_b / duration_a)) * 100
    print("\n=== 결과 요약 ===")
    print(f"성능 향상: {improvement:.2f}%")
    if duration_b < duration_a:
        print("✅ 성공: 임베딩 재사용을 통해 인덱싱 속도가 획기적으로 개선되었습니다.")
    else:
        print("❌ 실패: 성능 향상이 관찰되지 않았습니다.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
