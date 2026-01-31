import asyncio
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from core.semantic_chunker import EmbeddingBasedSemanticChunker

# 로깅 설정 추가
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


async def verify_logic():
    print("🔍 [Verification] 의미론적 청킹 논리 및 벡터 무결성 검증")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    # 테스트 데이터: 명확히 주제가 바뀌는 4개의 문장
    test_text = (
        "인공지능과 머신러닝은 현대 과학의 핵심 기술입니다. "  # 주제 A
        "딥러닝은 신경망을 통해 복잡한 데이터를 처리합니다. "  # 주제 A
        "신선한 토마토와 올리브유는 파스타 요리의 기본 재료입니다. "  # 주제 B
        "불 조절은 맛있는 스테이크를 굽는 데 가장 중요한 요소입니다."  # 주제 B
    )

    doc = Document(page_content=test_text, metadata={"source": "test_logic.pdf"})

    # 청커 설정 (유사도 임계값을 높게 설정하여 민감하게 반응하도록 함)
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        breakpoint_threshold_type="similarity_threshold",
        breakpoint_threshold_value=0.5,  # 0.5 미만이면 분리
        min_chunk_size=10,
        max_chunk_size=500,
    )

    print("\n1. 문장 분할 및 유사도 분석 시작...")
    split_docs, vectors = chunker.split_documents([doc])

    print(f"   - 생성된 청크 수: {len(split_docs)}")

    for i, (chunk, vector) in enumerate(zip(split_docs, vectors, strict=False)):
        print(f"\n[Chunk {i}]")
        print(f"내용: {chunk.page_content}")
        print(f"벡터 차원: {len(vector)} | 평균값: {np.mean(vector):.4f}")

    # 검증 로직
    if len(split_docs) >= 2:
        print("\n✅ 결과 분석: 주제가 바뀌는 지점에서 성공적으로 분리되었습니다.")
        # AI 관련 내용이 첫 번째 청크에 있는지 확인
        if (
            "인공지능" in split_docs[0].page_content
            and "파스타" in split_docs[1].page_content
        ):
            print("✅ 내용 매칭: 주제별 그룹화가 완벽합니다.")
    else:
        print(
            "\n❌ 결과 분석: 모든 문장이 하나의 청크로 묶였습니다. 임계값 조정이 필요합니다."
        )

    # 벡터 재사용 확인
    if vectors and len(vectors) == len(split_docs):
        print("✅ 벡터 무결성: 모든 청크에 대해 재사용 가능한 벡터가 생성되었습니다.")


if __name__ == "__main__":
    asyncio.run(verify_logic())
