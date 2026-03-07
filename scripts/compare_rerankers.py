
import asyncio
import logging
import time
import os
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
from langchain_core.documents import Document

# 테스트 환경 설정 (Mock 모드 강제)
os.environ["IS_UNIT_TEST"] = "true"

from core.reranker import DistributedReranker, RerankerStrategy
from core.model_loader import ModelManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RerankerBenchmark")

# 테스트용 Mock 데이터 (점수 포함)
class MockDoc:
    def __init__(self, content, page, doc_id, score):
        self.content = content
        self.metadata = {"page": page, "doc_id": doc_id}
        self.score = score
        self.doc_id = doc_id

MOCK_DOCS = [
    MockDoc("Chain-of-thought (CoT) prompting is a recently developed prompting method for improving reasoning.", 1, "1", 0.95),
    MockDoc("CoT prompting enables models to decompose complex problems into intermediate steps.", 1, "2", 0.94),
    MockDoc("We explore how chain-of-thought prompting improves performance on arithmetic tasks.", 2, "3", 0.85),
    MockDoc("Experiments show that CoT prompting significantly outperforms standard prompting on GSM8K.", 5, "4", 0.80),
    MockDoc("The effectiveness of CoT prompting is particularly evident in 100B+ models.", 7, "5", 0.75),
    MockDoc("Intermediate reasoning steps in CoT provide a window into the model's thought process.", 8, "6", 0.70),
    MockDoc("Chain-of-thought prompting does not require fine-tuning, making it accessible.", 10, "7", 0.65),
    MockDoc("However, CoT prompting might not be as effective for smaller models under 10B.", 12, "8", 0.60),
]

async def benchmark_rerankers():
    query = "How does chain-of-thought prompting improve large language models?"
    print(f"\n[1/3] Mock 데이터 준비 완료 ({len(MOCK_DOCS)}개 문서)")

    # 1. 임베딩 모델 (평가용 - Mock이 아닌 실제 CPU 모델 시도)
    # 텐서 오류 방지를 위해 임베딩 없이 텍스트 기반 다양성 지표 사용
    
    # 2. 전략 1: FlashRank (Baseline)
    print("\n[2/3] 전략 1 실행: FlashRank (TinyBERT)")
    ranker_flash = await ModelManager.get_flashranker()
    from flashrank import RerankRequest
    
    passages = [{"id": d.metadata["doc_id"], "text": d.page_content, "meta": d.metadata} for d in MOCK_DOCS]
    
    start_time = time.time()
    flash_results = ranker_flash.rerank(RerankRequest(query=query, passages=passages))
    flash_latency = time.time() - start_time
    flash_docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in flash_results[:4]]
    
    # 3. 전략 2: DistributedReranker - DIVERSITY (Proposed)
    print("[3/3] 전략 2 실행: DistributedReranker - DIVERSITY (MMR)")
    ranker_dist = DistributedReranker()
    
    start_time = time.time()
    dist_results, _ = ranker_dist.rerank(
        results=MOCK_DOCS,
        query_text=query,
        strategy=RerankerStrategy.DIVERSITY,
        top_k=4,
        diversity_weight=0.8
    )
    dist_latency = time.time() - start_time
    dist_docs = [Document(page_content=r.content, metadata=r.metadata) for r in dist_results]

    # 4. 결과 비교
    print("\n" + "="*70)
    print(f"{'Strategy':<20} | {'Latency':<10} | {'Top Pages (Diverse?)'}")
    print("-" * 70)
    
    flash_pages = [d.metadata["page"] for d in flash_docs]
    dist_pages = [d.metadata["page"] for d in dist_docs]
    
    print(f"{'FlashRank':<20} | {flash_latency:<10.4f} | {flash_pages}")
    print(f"{'DIVERSITY (MMR)':<20} | {dist_latency:<10.4f} | {dist_pages}")
    print("="*70)

    print("\n[FlashRank 결과 (순서대로)]")
    for i, d in enumerate(flash_docs):
        print(f" {i+1}. [P{d.metadata['page']}] {d.page_content[:100]}...")

    print("\n[DIVERSITY 결과 (다양성 고려)]")
    for i, d in enumerate(dist_docs):
        print(f" {i+1}. [P{d.metadata['page']}] {d.page_content[:100]}...")

    # 다양성 해석: DIVERSITY 전략은 같은 페이지(P1)에 몰려있는 결과보다 
    # 여러 페이지(P1, P5, P7, P12 등)에 흩어진 정보를 선호해야 합니다.

if __name__ == "__main__":
    asyncio.run(benchmark_rerankers())
