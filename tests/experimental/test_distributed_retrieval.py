import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from services.distributed.distributed_search import DistributedSearchManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DistTest")


async def test_distributed_flow():
    logger.info("=== 분산 검색 기능 테스트 시작 ===")

    # 1. 가상의 두 개 데이터셋 준비 (실제 문서 객체)
    docs_a = [
        Document(
            page_content="Apple is a fruit that is red or green.",
            metadata={"source": "NodeA", "page": 1},
        ),
        Document(
            page_content="Banana is a long yellow fruit.",
            metadata={"source": "NodeA", "page": 2},
        ),
    ]
    docs_b = [
        Document(
            page_content="The sun is a star at the center of the solar system.",
            metadata={"source": "NodeB", "page": 1},
        ),
        Document(
            page_content="Earth is the third planet from the sun.",
            metadata={"source": "NodeB", "page": 2},
        ),
    ]

    # 2. 개별 노드 리트리버 생성
    retriever_a = BM25Retriever.from_documents(docs_a)
    retriever_b = BM25Retriever.from_documents(docs_b)

    # 3. 분산 매니저 설정
    manager = DistributedSearchManager()
    manager.add_node("FruitNode", retriever_a)
    manager.add_node("SpaceNode", retriever_b)

    dist_retriever = manager.get_retriever(top_k=2)

    # 4. 검색 테스트
    query = "Tell me about the sun and fruit"
    logger.info(f"질문: '{query}'")

    results = await dist_retriever.ainvoke(query)

    logger.info(f"검색 결과 ({len(results)}개):")
    for i, doc in enumerate(results):
        logger.info(f"[{i + 1}] [{doc.metadata['source']}] {doc.page_content}")

    # 5. 검증
    sources = [doc.metadata["source"] for doc in results]
    if "NodeA" in sources and "NodeB" in sources:
        logger.info("✅ 성공: 여러 노드에서 결과가 올바르게 병합되었습니다.")
    else:
        logger.warning("❌ 실패: 모든 노드의 결과가 포함되지 않았습니다.")


if __name__ == "__main__":
    asyncio.run(test_distributed_flow())
