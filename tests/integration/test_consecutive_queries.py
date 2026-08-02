import asyncio
import os
import sys

import pytest

# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from core.model_loader import load_embedding_model
from core.rag_core import RAGSystem


@pytest.mark.asyncio
async def test_consecutive_queries():
    """
    동일한 RAGSystem 인스턴스에 대해 연속적으로 2개의 질문을 수행하여
    시스템 안정성과 답변의 정확성을 검증합니다.
    """
    print("\n🚀 연속 질문 통합 테스트 시작...")

    # 1. 모델 및 시스템 준비
    embedding_model = AVAILABLE_EMBEDDING_MODELS[0]
    llm_model = DEFAULT_OLLAMA_MODEL

    print(f"🧬 모델 로딩: {embedding_model}, {llm_model}")
    embedder = await asyncio.to_thread(load_embedding_model, embedding_model)

    rag = RAGSystem(session_id="consecutive_test_session")

    pdf_path = os.path.join("tests", "data", "2201.07520v1.pdf")
    print(f"📂 문서 인덱싱: {pdf_path}")
    await rag.build_pipeline(
        file_path=pdf_path,
        file_name="2201.07520v1.pdf",
        embedder=embedder,
    )

    # 2. 첫 번째 질문: 개요 위주
    q1 = "What is the main objective of the CM3 model?"
    print(f"\n💬 질문 1: {q1}")
    res1 = await rag.aquery(q1, model_name=llm_model)
    ans1 = res1.get("response", "")

    print("-" * 30)
    print(f"🤖 답변 1:\n{ans1}")
    print("-" * 30)

    assert len(ans1) > 50, "첫 번째 답변이 너무 짧거나 비어있습니다."
    assert "CM3" in ans1 or "causally masked" in ans1.lower(), (
        "질문 1에 대한 답변이 부적절합니다."
    )

    # 3. 두 번째 질문: 세부 사항 위주 (메모리 및 컨텍스트 확인)
    q2 = "What datasets were used to train the CM3 models?"
    print(f"\n💬 질문 2: {q2}")
    res2 = await rag.aquery(q2, model_name=llm_model)
    ans2 = res2.get("response", "")

    print("-" * 30)
    print(f"🤖 답변 2:\n{ans2}")
    print("-" * 30)

    assert len(ans2) > 50, "두 번째 답변이 너무 짧거나 비어있습니다."
    # 논문 내용상 Wikipedia, Common Crawl 등이 언급되는지 확인 (실제 답변에 따라 유연하게 검증)
    assert any(
        word in ans2.lower()
        for word in ["data", "web", "wikipedia", "common", "dataset"]
    ), "질문 2에 대한 답변에 데이터 관련 키워드가 부족합니다."

    print("\n✅ 두 질문 모두 성공적으로 처리되었습니다.")
    print(f"질문 1 참조 문서 수: {len(res1.get('documents', []))}")
    print(f"질문 2 참조 문서 수: {len(res2.get('documents', []))}")
    print("\n✨ 연속 질문 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_consecutive_queries())
