"""
벡터 DB 최적화 모듈 - 배치 인덱싱, 병렬 검색, Re-ranking 파이프라인.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Coroutine
from collections import defaultdict
import numpy as np
from threading import RLock

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexingStrategy(Enum):
    """인덱싱 전략."""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    PARALLEL = "parallel"


class SearchStrategy(Enum):
    """검색 전략."""
    SINGLE_QUERY = "single_query"
    PARALLEL_QUERIES = "parallel_queries"
    MULTI_VECTOR = "multi_vector"


@dataclass
class BatchIndexConfig:
    """배치 인덱싱 설정."""
    batch_size: int = 32
    max_workers: int = 4
    timeout_per_batch: float = 30.0
    retry_failed: bool = True
    max_retries: int = 3
    enable_deduplication: bool = True
    compression_enabled: bool = False


@dataclass
class ParallelSearchConfig:
    """병렬 검색 설정."""
    max_concurrent_searches: int = 4
    timeout_per_search: float = 10.0
    k_results: int = 5
    score_threshold: Optional[float] = None


@dataclass
class RerankingConfig:
    """Re-ranking 설정."""
    enable_reranking: bool = True
    rerank_model: Optional[str] = None
    top_k_before_rerank: int = 20
    top_k_after_rerank: int = 5
    diversity_penalty: float = 0.0
    relevance_weight: float = 0.7


@dataclass
class BatchIndexResult:
    """배치 인덱싱 결과."""
    total_indexed: int = 0
    successful: int = 0
    failed: int = 0
    duplicates_removed: int = 0
    execution_time: float = 0.0
    failed_indices: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """검색 결과."""
    query: str
    results: List[Tuple[Document, float]]
    execution_time: float
    num_candidates_before_rerank: Optional[int] = None
    num_results_after_rerank: Optional[int] = None
    rerank_scores: Optional[List[float]] = None


class DocumentDeduplicator:
    """문서 중복 제거기."""
    
    def __init__(self, hash_func: Optional[Callable] = None):
        """
        Args:
            hash_func: 문서 해시 함수 (기본값: content 기반)
        """
        self.hash_func = hash_func or self._default_hash
        self.seen_hashes: Dict[str, Document] = {}
        self._lock = RLock()
    
    def _default_hash(self, doc: Document) -> str:
        """문서 콘텐츠 기반 해시."""
        import hashlib
        content = doc.page_content
        metadata_str = str(sorted(doc.metadata.items()))
        combined = content + metadata_str
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def add_document(self, doc: Document) -> Tuple[bool, str]:
        """
        문서 추가 (중복이면 무시).
        
        Returns:
            (추가됨 여부, 해시값)
        """
        with self._lock:
            doc_hash = self.hash_func(doc)
            if doc_hash in self.seen_hashes:
                return False, doc_hash
            self.seen_hashes[doc_hash] = doc
            return True, doc_hash
    
    def deduplicate_batch(self, documents: List[Document]) -> Tuple[List[Document], int]:
        """
        배치 문서 중복 제거.
        
        Returns:
            (중복 제거된 문서 리스트, 제거된 개수)
        """
        unique_docs = []
        removed_count = 0
        
        for doc in documents:
            added, _ = self.add_document(doc)
            if added:
                unique_docs.append(doc)
            else:
                removed_count += 1
        
        return unique_docs, removed_count
    
    def clear(self):
        """중복 제거기 초기화."""
        with self._lock:
            self.seen_hashes.clear()


class BatchIndexer:
    """배치 인덱싱 관리자."""
    
    def __init__(self, config: BatchIndexConfig):
        self.config = config
        self.deduplicator = DocumentDeduplicator() if config.enable_deduplication else None
        self._lock = RLock()
    
    async def batch_index_documents(
        self,
        documents: List[Document],
        index_func: Callable[[List[Document]], Coroutine[Any, Any, Any]],
    ) -> BatchIndexResult:
        """
        배치로 문서 인덱싱.
        
        Args:
            documents: 인덱싱할 문서 리스트
            index_func: 배치 인덱싱 함수 (배치 크기 리스트를 받아 처리)
        
        Returns:
            배치 인덱싱 결과
        """
        import time
        start_time = time.time()
        result = BatchIndexResult(total_indexed=len(documents))
        
        # 중복 제거
        if self.deduplicator:
            unique_docs, duplicates = self.deduplicator.deduplicate_batch(documents)
            result.duplicates_removed = duplicates
            documents = unique_docs
        
        # 배치 처리
        batches = [
            documents[i:i + self.config.batch_size]
            for i in range(0, len(documents), self.config.batch_size)
        ]
        
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._index_batch_with_retry(batch, batch_idx, index_func)
            tasks.append(task)
        
        # 동시 처리
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 집계
        for batch_idx, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                result.failed += len(batches[batch_idx])
                result.failed_indices.append(batch_idx)
                result.errors.append(str(batch_result))
            else:
                result.successful += batch_result.get("indexed", 0)
                if batch_result.get("failed", 0) > 0:
                    result.failed += batch_result.get("failed", 0)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _index_batch_with_retry(
        self,
        batch: List[Document],
        batch_idx: int,
        index_func: Callable,
        retry_count: int = 0,
    ) -> Dict[str, int]:
        """배치 인덱싱 (재시도 포함)."""
        try:
            # 실제 인덱싱 함수 호출
            result = await asyncio.wait_for(
                index_func(batch),
                timeout=self.config.timeout_per_batch
            )
            return {"indexed": len(batch), "failed": 0}
        except asyncio.TimeoutError:
            if retry_count < self.config.max_retries and self.config.retry_failed:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._index_batch_with_retry(
                    batch, batch_idx, index_func, retry_count + 1
                )
            raise
        except Exception as e:
            logger.error(f"배치 {batch_idx} 인덱싱 실패: {e}")
            raise


class ParallelSearcher:
    """병렬 검색 관리자."""
    
    def __init__(self, config: ParallelSearchConfig):
        self.config = config
        self._lock = RLock()
    
    async def search_parallel(
        self,
        queries: List[str],
        search_func: Callable[[str, int], Coroutine[Any, Any, List[Tuple[Document, float]]]],
    ) -> List[SearchResult]:
        """
        다중 쿼리 병렬 검색.
        
        Args:
            queries: 검색 쿼리 리스트
            search_func: 쿼리와 k를 받아서 (doc, score) 튜플 리스트 반환
        
        Returns:
            검색 결과 리스트
        """
        import time
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        
        async def search_with_semaphore(query: str) -> SearchResult:
            async with semaphore:
                return await self._search_single_query(
                    query, search_func
                )
        
        tasks = [search_with_semaphore(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 에러 처리
        search_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"검색 실패: {result}")
            else:
                search_results.append(result)
        
        return search_results
    
    async def _search_single_query(
        self,
        query: str,
        search_func: Callable,
    ) -> SearchResult:
        """단일 쿼리 검색."""
        import time
        start_time = time.time()
        
        try:
            results = await asyncio.wait_for(
                search_func(query, self.config.k_results),
                timeout=self.config.timeout_per_search
            )
            
            # Score filtering
            if self.config.score_threshold:
                results = [
                    (doc, score) for doc, score in results
                    if score >= self.config.score_threshold
                ]
            
            execution_time = time.time() - start_time
            return SearchResult(
                query=query,
                results=results,
                execution_time=execution_time,
                num_candidates_before_rerank=len(results),
            )
        except asyncio.TimeoutError:
            logger.warning(f"검색 타임아웃: {query}")
            raise


class Reranker:
    """Re-ranking 엔진."""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self._lock = RLock()
    
    def rerank_results(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        rerank_func: Optional[Callable] = None,
    ) -> Tuple[List[Tuple[Document, float]], List[float]]:
        """
        검색 결과 Re-ranking.
        
        Args:
            query: 원본 쿼리
            results: (문서, 스코어) 튜플 리스트
            rerank_func: 커스텀 re-ranking 함수
        
        Returns:
            (Re-ranked 결과, Re-ranking 스코어)
        """
        if not self.config.enable_reranking:
            return results, [score for _, score in results]
        
        # 상위 K개 후보 선택
        top_candidates = results[:self.config.top_k_before_rerank]
        
        if rerank_func:
            # 커스텀 re-ranking 함수 사용
            rerank_scores = rerank_func(query, top_candidates)
        else:
            # 기본 re-ranking: 다양성 패널티 + 관련도
            rerank_scores = self._compute_rerank_scores(query, top_candidates)
        
        # 스코어로 정렬
        ranked = sorted(
            zip(top_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 최종 결과
        final_results = [
            (doc, score) for (doc, orig_score), score in ranked[:self.config.top_k_after_rerank]
        ]
        final_scores = [score for _, score in final_results]
        
        return final_results, final_scores
    
    def _compute_rerank_scores(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
    ) -> List[float]:
        """기본 re-ranking 스코어 계산."""
        scores = []
        
        # 문서 간 다양성 계산
        for i, (doc_i, orig_score_i) in enumerate(candidates):
            diversity_penalty = 0.0
            
            # 이전 문서와의 유사도 계산
            for j in range(i):
                doc_j, _ = candidates[j]
                # 간단한 유사도: 공통 단어 수 / 총 단어 수
                words_i = set(doc_i.page_content.lower().split())
                words_j = set(doc_j.page_content.lower().split())
                
                if words_i and words_j:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    similarity = intersection / union if union > 0 else 0
                    diversity_penalty += similarity * self.config.diversity_penalty
            
            # 최종 스코어
            final_score = (
                self.config.relevance_weight * orig_score_i +
                (1 - self.config.relevance_weight) * (1 - diversity_penalty)
            )
            scores.append(final_score)
        
        return scores


class VectorDBOptimizer:
    """벡터 DB 최적화 통합 관리자."""
    
    def __init__(
        self,
        batch_config: Optional[BatchIndexConfig] = None,
        search_config: Optional[ParallelSearchConfig] = None,
        rerank_config: Optional[RerankingConfig] = None,
    ):
        self.batch_config = batch_config or BatchIndexConfig()
        self.search_config = search_config or ParallelSearchConfig()
        self.rerank_config = rerank_config or RerankingConfig()
        
        self.batch_indexer = BatchIndexer(self.batch_config)
        self.parallel_searcher = ParallelSearcher(self.search_config)
        self.reranker = Reranker(self.rerank_config)
        
        self._lock = RLock()
        self._metrics = {
            "total_indexed": 0,
            "total_searches": 0,
            "avg_search_time": 0.0,
            "total_reranked": 0,
        }
    
    async def optimize_indexing(
        self,
        documents: List[Document],
        index_func: Callable[[List[Document]], Coroutine[Any, Any, Any]],
    ) -> BatchIndexResult:
        """최적화된 배치 인덱싱."""
        result = await self.batch_indexer.batch_index_documents(documents, index_func)
        
        with self._lock:
            self._metrics["total_indexed"] += result.successful
        
        return result
    
    async def optimize_search(
        self,
        queries: List[str],
        search_func: Callable,
        rerank_func: Optional[Callable] = None,
    ) -> List[SearchResult]:
        """최적화된 병렬 검색 + Re-ranking."""
        # 병렬 검색
        search_results = await self.parallel_searcher.search_parallel(
            queries, search_func
        )
        
        # Re-ranking
        reranked_results = []
        for result in search_results:
            final_results, rerank_scores = self.reranker.rerank_results(
                result.query,
                result.results,
                rerank_func
            )
            result.results = final_results
            result.rerank_scores = rerank_scores
            result.num_results_after_rerank = len(final_results)
            reranked_results.append(result)
        
        with self._lock:
            self._metrics["total_searches"] += len(queries)
            avg_time = sum(r.execution_time for r in reranked_results) / len(reranked_results)
            self._metrics["avg_search_time"] = avg_time
            self._metrics["total_reranked"] += sum(
                r.num_results_after_rerank or 0 for r in reranked_results
            )
        
        return reranked_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """최적화 메트릭 조회."""
        with self._lock:
            return self._metrics.copy()


# 전역 인스턴스
_optimizer_instance: Optional[VectorDBOptimizer] = None


def get_vector_db_optimizer() -> VectorDBOptimizer:
    """전역 벡터 DB 최적화 인스턴스 조회."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = VectorDBOptimizer()
    return _optimizer_instance


def reset_vector_db_optimizer():
    """벡터 DB 최적화 인스턴스 리셋 (테스트용)."""
    global _optimizer_instance
    _optimizer_instance = None
