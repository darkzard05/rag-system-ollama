import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.graph_builder import preprocess
from langchain_core.runnables import RunnableConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_dynamic_weights():
    """동적 가중치 할당 로직을 검증합니다."""
    
    test_cases = [
        {
            "query": "v1.5.0 버전의 주요 변경 사항은?",
            "expected_intent": "keyword",
            "desc": "버전 정보 포함 (BM25 강화 예상)"
        },
        {
            "query": "RAG 시스템과 표준 검색의 차이점이 뭐야?",
            "expected_intent": "semantic",
            "desc": "개념 비교 질의 (FAISS 강화 예상)"
        },
        {
            "query": "안녕?",
            "expected_intent": "general",
            "desc": "일상 대화"
        }
    ]

    mock_writer = lambda x: None
    mock_config = RunnableConfig(configurable={})

    logger.info("=== 동적 가중치 할당 테스트 시작 ===")
    
    for case in test_cases:
        state = {"input": case["query"], "is_cached": False, "retry_count": 0}
        result = await preprocess(state, mock_config, mock_writer)
        
        weights = result.get("search_weights", {})
        intent = result.get("intent")
        
        logger.info(f"\n[질문]: {case['query']} ({case['desc']})")
        logger.info(f"- 결과 의도: {intent}")
        logger.info(f"- 할당 가중치: BM25({weights.get('bm25', 'N/A')}), FAISS({weights.get('faiss', 'N/A')})")
        
        if case["expected_intent"] == "keyword":
            assert weights.get("bm25") > weights.get("faiss"), "키워드 중심 질의에서 BM25 비중이 낮음"
        elif case["expected_intent"] == "semantic":
            assert weights.get("faiss") > weights.get("bm25"), "의미 중심 질의에서 FAISS 비중이 낮음"

    logger.info("\n✅ 모든 동적 가중치 할당 테스트 통과!")

if __name__ == "__main__":
    asyncio.run(verify_dynamic_weights())
