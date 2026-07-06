import sys
import os

# src 디렉토리를 path의 가장 앞에 추가
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import logging
from langchain_core.documents import Document
from src.core.reranker import FlashReranker


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RerankerTest")


def test_reranker_reliability():
    # 1. 리랭커 초기화
    logger.info("Initializing FlashReranker...")
    reranker = FlashReranker()

    if not reranker.ranker:
        logger.error(
            "Reranker failed to initialize. Please check if flashrank is installed."
        )
        return

    # 2. 테스트 시나리오 정의
    # 질의: 프로젝트의 메인 진입점과 그 역할에 대해 질문
    query = "프로젝트의 메인 진입점 파일과 그 역할은 무엇인가요?"

    test_docs = [
        Document(
            page_content="src/main.py는 Streamlit 기반 RAG 챗봇의 메인 진입점이며, UI 구성과 세션 상태 관리 및 전체 오케스트레이션을 담당하는 파일입니다.",
            metadata={"id": "doc_high", "label": "Highly Relevant"},
        ),
        Document(
            page_content="RAG(Retrieval-Augmented Generation) 시스템은 외부 지식 베이스에서 관련 문서를 검색하여 LLM의 답변 정확도를 높이는 기술입니다.",
            metadata={"id": "doc_mid", "label": "Partially Relevant"},
        ),
        Document(
            page_content="오늘의 날씨는 매우 화창하며, 전국적으로 미세먼지 농도가 낮아 야외 활동을 하기 좋은 날씨입니다.",
            metadata={"id": "doc_low", "label": "Irrelevant"},
        ),
        Document(
            page_content="python의 list comprehension은 코드를 간결하게 만들어주며, 반복문을 효율적으로 처리하는 방법 중 하나입니다.",
            metadata={"id": "doc_low2", "label": "Irrelevant"},
        ),
    ]

    logger.info(f"\nQuery: {query}")
    logger.info("-" * 50)

    # 3. 리랭킹 수행
    # top_k를 넉넉하게 설정하여 모든 문서의 점수를 확인
    results = reranker.rerank_documents(query, test_docs, top_k=len(test_docs))

    # 4. 결과 분석 및 출력
    print(f"\n{'Label':<20} | {'Score':<10} | {'Content'}")
    print("-" * 80)

    for doc in results:
        label = "Unknown"
        # 원본 문서 리스트에서 레이블 찾기
        for td in test_docs:
            if td.page_content == doc.page_content:
                label = td.metadata.get("label", "Unknown")
                break

        score = doc.metadata.get("rerank_score", 0.0)
        content = (
            doc.page_content[:50] + "..."
            if len(doc.page_content) > 50
            else doc.page_content
        )
        print(f"{label:<20} | {score:<10.4f} | {content}")

    # 5. 결론 도출을 위한 분석
    scores = [doc.metadata.get("rerank_score", 0.0) for doc in results]
    max_score = max(scores)
    min_score = min(scores)

    print("-" * 80)
    print(f"Max Score: {max_score:.4f}")
    print(f"Min Score: {min_score:.4f}")
    print(f"Score Gap (Max-Min): {max_score - min_score:.4f}")


if __name__ == "__main__":
    test_reranker_reliability()
