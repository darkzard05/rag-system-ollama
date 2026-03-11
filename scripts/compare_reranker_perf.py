
import asyncio
import time
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from core.reranker import DistributedReranker, RerankerStrategy

def keyword_overlap_score(query, text):
    """이전 방식: 단순 단어 겹침 점수 계산"""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words: return 0.0
    return len(q_words & t_words) / len(q_words)

async def run_benchmark():
    reranker = DistributedReranker()
    # 리랭킹 강제를 위해 threshold 상향
    reranker.bypass_threshold = 1.0 

    test_cases = [
        {
            "query": "대한민국의 수도는 어디인가요?",
            "docs": [
                Document(page_content="서울은 한국의 중심지이자 행정 수도입니다.", metadata={"id": "GT"}), # Ground Truth
                Document(page_content="수도꼭지에서 물이 나옵니다.", metadata={"id": "Noise"}), # 단어는 겹치지만 오답
                Document(page_content="오늘 날씨는 맑습니다.", metadata={"id": "Irrelevant"})
            ]
        },
        {
            "query": "아이폰 액정 수리 비용",
            "docs": [
                Document(page_content="애플 스마트폰 디스플레이 교체 서비스 안내", metadata={"id": "GT"}), # 의미 일치
                Document(page_content="아이폰 케이스 판매합니다.", metadata={"id": "Noise"}), # 단어 일치, 맥락 오답
                Document(page_content="삼성 갤럭시 수리 센터 위치", metadata={"id": "Irrelevant"})
            ]
        }
    ]

    print(f"{'전략':<15} | {'정확도(P@1)':<10} | {'평균 지연시간':<15}")
    print("-" * 50)

    # 1. Previous (Keyword Overlap)
    start = time.time()
    correct_count = 0
    for case in test_cases:
        # 키워드 점수 기반 수동 정렬
        sorted_docs = sorted(case["docs"], key=lambda x: keyword_overlap_score(case["query"], x.page_content), reverse=True)
        if sorted_docs[0].metadata["id"] == "GT":
            correct_count += 1
    
    latency = (time.time() - start) / len(test_cases)
    print(f"{'이전(Keyword)':<15} | {correct_count/len(test_cases):<12.1%} | {latency*1000:>12.2f}ms")

    # 2. Current (FlashRank Semantic)
    start = time.time()
    correct_count = 0
    for case in test_cases:
        final_docs, metrics = reranker.rerank(case["docs"], query_text=case["query"], strategy=RerankerStrategy.SEMANTIC_FLASH)
        if final_docs[0].metadata["id"] == "GT":
            correct_count += 1
    
    latency = (time.time() - start) / len(test_cases)
    print(f"{'현재(FlashRank)':<15} | {correct_count/len(test_cases):<12.1%} | {latency*1000:>12.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
