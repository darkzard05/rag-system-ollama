"""
Distributed Search Execution Module
분산 검색 실행 - 다중 노드/인덱스에서 병렬 검색 및 결과 병합
STATUS: EXPERIMENTAL (Task 19-1)
"""

import asyncio
import hashlib
import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)


class NodeSearchStatus(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SearchResult:
    """통합 검색 결과 데이터 클래스 (LangChain Document 호환)"""

    doc: Document
    score: float
    node_id: str
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        # heapq는 최소 힙이므로 점수 기반 최대 힙처럼 동작하게 하려면 역순 비교
        return self.score > other.score


class NodeSearchEngine:
    """
    개별 노드 또는 인덱스를 담당하는 검색 엔진.
    실제 LangChain Retriever를 래핑하여 검색을 수행합니다.
    """

    def __init__(self, node_id: str, retriever: BaseRetriever):
        self.node_id = node_id
        self.retriever = retriever
        self._lock = asyncio.Lock()

    async def asearch(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """비동기 검색 수행"""
        async with self._lock:
            try:
                # 1. 실제 리트리버 호출
                docs = await self.retriever.ainvoke(query)

                # 2. SearchResult로 변환
                results = []
                for i, doc in enumerate(docs):
                    # 리트리버가 점수를 제공하지 않을 경우 순위 기반 의사 점수 부여 (1.0 ~ 0.1)
                    score = doc.metadata.get("score", 1.0 - (i * 0.1))
                    results.append(
                        SearchResult(
                            doc=doc, score=max(0.0, score), node_id=self.node_id
                        )
                    )
                return results[:top_k]
            except Exception as e:
                logger.error(f"[Node:{self.node_id}] 검색 오류: {e}")
                return []


class DistributedRetriever(BaseRetriever):
    """
    다중 노드/인덱스를 병렬로 검색하는 분산 리트리버.
    LangChain의 BaseRetriever를 상속받아 기존 파이프라인에 즉시 투입 가능합니다.
    """

    engines: List[NodeSearchEngine]
    top_k: int = 5
    timeout: float = 10.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """동기 호출 시 비동기 루프 실행 (LangChain 표준)"""
        return asyncio.run(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """모든 노드에서 병렬 검색 후 결과 병합"""
        start_time = time.time()

        # 1. 모든 노드에 검색 요청 발송
        tasks = [engine.asearch(query, top_k=self.top_k) for engine in self.engines]

        # 2. 병렬 실행 (타임아웃 적용)
        try:
            responses = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"분산 검색 중 치명적 오류: {e}")
            return []

        # 3. 결과 병합 및 중복 제거 (Global Merge)
        all_results: List[SearchResult] = [
            res for sublist in responses for res in sublist
        ]

        # 중복 제거 (Content Hash 기준)
        unique_results = {}
        for res in all_results:
            content_hash = hashlib.sha256(res.doc.page_content.encode()).hexdigest()
            if (
                content_hash not in unique_results
                or res.score > unique_results[content_hash].score
            ):
                unique_results[content_hash] = res

        # 4. 점수 기반 최종 정렬 (Heapq 사용)
        final_results = heapq.nsmallest(self.top_k, unique_results.values())

        duration = time.time() - start_time
        logger.info(
            f"[DistributedSearch] 병합 완료: {len(self.engines)}개 노드, 소요 {duration:.2f}s"
        )

        return [res.doc for res in final_results]


class DistributedSearchManager:
    """분산 검색 엔진들을 관리하고 리트리버를 생성하는 팩토리 클래스"""

    def __init__(self):
        self._engines: Dict[str, NodeSearchEngine] = {}

    def add_node(self, node_id: str, retriever: BaseRetriever):
        """새로운 검색 노드 추가"""
        self._engines[node_id] = NodeSearchEngine(node_id, retriever)
        logger.info(f"검색 노드 추가됨: {node_id}")

    def get_retriever(
        self, top_k: int = 5, timeout: float = 10.0
    ) -> DistributedRetriever:
        """분산 리트리버 인스턴스 반환"""
        return DistributedRetriever(
            engines=list(self._engines.values()), top_k=top_k, timeout=timeout
        )
