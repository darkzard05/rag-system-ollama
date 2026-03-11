
import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from core.reranker import DistributedReranker, RerankerStrategy
from common.config import RERANKER_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async def test_reranker():
    print("\n=== [1] FlashRank 초기화 및 설정 로드 테스트 ===")
    reranker = DistributedReranker()
    print(f"현재 설정된 모델: {RERANKER_CONFIG.get('model_name')}")
    print(f"Bypass Threshold: {reranker.bypass_threshold}")
    print(f"Default Top K: {reranker.default_top_k}")

    # 가상 검색 결과 생성 (점수는 높지만 질문과 관련성 낮은 문서 vs 점수는 낮지만 관련성 높은 문서)
    query = "대한민국의 수도는 어디인가요?"
    mock_results = [
        Document(
            page_content="오늘의 날씨는 매우 맑습니다. 기온은 20도입니다.", 
            metadata={"score": 0.9} # 높은 점수 (하지만 질문과 무관)
        ),
        Document(
            page_content="서울은 대한민국의 수도이며 정치, 경제의 중심지입니다.", 
            metadata={"score": 0.4} # 낮은 점수 (하지만 질문과 밀접)
        ),
        Document(
            page_content="피자는 이탈리아에서 유래된 음식입니다.", 
            metadata={"score": 0.1}
        )
    ]

    print("\n=== [2] Bypass 로직 테스트 (High Score) ===")
    # Bypass 테스트를 위해 임계치 조정된 리랭커
    reranker.bypass_threshold = 0.8 
    final_docs, metrics = reranker.rerank(mock_results, query_text=query)
    print(f"Bypass 결과: {metrics.strategy_used} (전략), {metrics.reranked_results}개 결과")
    if metrics.reranking_time < 0.001:
        print("성공: 높은 점수의 문서가 있어 리랭킹을 건너뛰었습니다.")

    print("\n=== [3] 시맨틱 리랭킹 테스트 (Semantic Re-ordering) ===")
    # 리랭킹을 강제하기 위해 threshold를 0.99로 상향
    reranker.bypass_threshold = 0.99
    final_docs, metrics = reranker.rerank(mock_results, query_text=query, top_k=2)
    
    print(f"리랭킹 소요 시간: {metrics.reranking_time:.4f}s")
    print(f"선별된 최상위 문서: {final_docs[0].page_content[:40]}...")
    
    # 두 번째 문서(서울 관련)가 첫 번째로 올라왔는지 확인
    if "서울" in final_docs[0].page_content:
        print("성공: FlashRank가 시맨틱하게 더 관련성 높은 문서를 상단으로 재배치했습니다.")
    else:
        print("실패: 문서 순위가 조정되지 않았습니다.")
        
    print(f"결과 개수 확인: {len(final_docs)} (Expected: 2)")

if __name__ == "__main__":
    asyncio.run(test_reranker())
