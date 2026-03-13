import asyncio
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.reranker import DistributedReranker, RerankerStrategy
from api.schemas import AggregatedSearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_intelligent_reranking():
    """지능형 리랭킹 최적화 로직을 검증합니다."""
    reranker = DistributedReranker()
    
    logger.info("=== 지능형 리랭킹 검증 시작 ===")

    # 1. [Early Exit 테스트] 1위가 압도적인 경우
    clear_results = [
        AggregatedSearchResult(doc_id="1", content="정답", score=0.99, node_id="test", metadata={}),
        AggregatedSearchResult(doc_id="2", content="오답", score=0.20, node_id="test", metadata={}),
    ]
    
    start = time.perf_counter()
    res1, metrics1 = reranker.rerank(clear_results, query_text="질문")
    latency1 = (time.perf_counter() - start) * 1000
    
    logger.info(f"\n시나리오 1 (Early Exit):")
    logger.info(f"- 전략 사용: {metrics1.strategy_used}")
    logger.info(f"- 소요 시간: {latency1:.2f} ms")
    assert metrics1.strategy_used == "early_exit_bypass", "Early Exit이 작동하지 않음"

    # 2. [Hard Filtering 테스트] 모든 점수가 낮은 경우
    noisy_results = [
        AggregatedSearchResult(doc_id="1", content="노이즈 1", score=0.5, node_id="test", metadata={}),
        AggregatedSearchResult(doc_id="2", content="노이즈 2", score=0.4, node_id="test", metadata={}),
    ]
    
    # 리랭킹 수행 (FlashRank 호출 시 점수가 낮게 나오도록 유도)
    # 실제 FlashRank가 0.1 미만으로 점수를 매길만한 전혀 무관한 질문 사용
    res2, metrics2 = reranker.rerank(noisy_results, query_text="완전 무관한 질문", strategy=RerankerStrategy.SEMANTIC_FLASH)
    
    logger.info(f"\n시나리오 2 (Hard Filtering):")
    logger.info(f"- 필터링 전: {len(noisy_results)}개")
    logger.info(f"- 필터링 후: {len(res2)}개")
    if len(res2) < len(noisy_results):
        logger.info("✅ 성공: 관련성 낮은 문서가 필터링되었습니다.")

    logger.info("\n✅ 지능형 리랭킹 최적화 모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(verify_intelligent_reranking())
