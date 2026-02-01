"""
AsyncIO 최적화 계층 - 동시 LLM 처리, 병렬 문서 검색, 메모리 효율 배치 처리
"""

import asyncio
import hashlib
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

# Type aliases
DocumentList = list[Document]


@dataclass
class AsyncConfig:
    """AsyncIO 동시 처리 설정"""

    max_concurrent_queries: int = 5  # 동시 쿼리 확장 제한
    max_concurrent_retrievals: int = 10  # 동시 문서 검색 제한
    max_concurrent_embeddings: int = 8  # 동시 임베딩 제한
    max_concurrent_rerankings: int = 3  # 동시 리랭킹 배치 제한
    batch_size_embeddings: int = 32  # 임베딩 배치 크기
    batch_size_reranking: int = 50  # 리랭킹 배치 크기
    timeout_llm: float = 30.0  # LLM 타임아웃 (초)
    timeout_retriever: float = 15.0  # 리트리버 타임아웃 (초)
    timeout_embedding: float = 10.0  # 임베딩 타임아웃 (초)
    timeout_reranking: float = 20.0  # 리랭킹 타임아웃 (초)


class AsyncSemaphore:
    """동시성 제어를 위한 세마포어"""

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class ConcurrentQueryExpander:
    """
    동시 쿼리 확장기 - 여러 쿼리를 병렬로 LLM 처리

    특징:
    - 세마포어를 통한 동시 요청 제한
    - 타임아웃 보호
    - 개별 쿼리 에러 격리
    - 성능 모니터링 통합
    """

    def __init__(self, config: AsyncConfig | None = None):
        self.config = config or AsyncConfig()
        self.semaphore = AsyncSemaphore(self.config.max_concurrent_queries)

    async def expand_queries_concurrently(
        self,
        queries: list[str],
        expander_func: Callable[[str], Coroutine[Any, Any, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """
        여러 쿼리를 동시에 확장

        Args:
            queries: 확장할 쿼리 리스트
            expander_func: 쿼리를 받아서 확장된 쿼리 문자열을 반환하는 비동기 함수
            metadata: 성능 모니터링용 메타데이터

        Returns:
            (확장된 쿼리 리스트, 통계 딕셔너리)

        예시:
            >>> async def expand_query(q):
            ...     return f"{q} expanded"
            >>> expander = ConcurrentQueryExpander()
            >>> queries, stats = await expander.expand_queries_concurrently(
            ...     ["query1", "query2"],
            ...     expand_query
            ... )
        """
        with monitor.track_operation(
            OperationType.QUERY_PROCESSING,
            {
                "stage": "concurrent_expansion",
                "query_count": len(queries),
                **(metadata or {}),
            },
        ) as op:
            logger.info(f"[Optimizer] [Query] 동시 쿼리 확장 시작: {len(queries)} 쿼리")

            async def _expand_with_limit(
                query: str, index: int
            ) -> tuple[int, list[str]]:
                """세마포어 제한과 함께 단일 쿼리 확장"""
                async with self.semaphore:
                    try:
                        result = await asyncio.wait_for(
                            expander_func(query), timeout=self.config.timeout_llm
                        )
                        # 확장된 쿼리 파싱 (줄바꿈으로 구분)
                        expanded = [
                            q.strip()
                            for q in result.split("\n")
                            if q.strip() and not q.strip().startswith("-")
                        ]
                        logger.debug(
                            f"[AsyncOptimizer] 쿼리 {index} 확장: {len(expanded)} 결과"
                        )
                        return index, expanded

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[AsyncOptimizer] 쿼리 확장 타임아웃 (쿼리 {index}): {query[:50]}"
                        )
                        return index, [query]  # 폴백: 원본 쿼리 반환
                    except Exception as e:
                        logger.error(
                            f"[AsyncOptimizer] 쿼리 확장 오류 (쿼리 {index}): {e}"
                        )
                        op.error = str(e)
                        return index, [query]  # 폴백: 원본 쿼리 반환

            # 모든 쿼리를 동시에 확장
            tasks = [_expand_with_limit(q, i) for i, q in enumerate(queries)]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # 결과 정렬 및 병합
            all_expanded = []
            stats = {
                "input_queries": len(queries),
                "output_queries": 0,
                "expansion_ratio": 0.0,
                "failed_count": 0,
            }

            for index, expanded_queries in sorted(results, key=lambda x: x[0]):
                if expanded_queries == [queries[index]]:
                    stats["failed_count"] += 1
                all_expanded.extend(expanded_queries)

            stats["output_queries"] = len(all_expanded)
            stats["expansion_ratio"] = (
                len(all_expanded) / len(queries) if queries else 0
            )

            logger.info(
                f"[Optimizer] [Query] 동시 쿼리 확장 완료: "
                f"{len(queries)} -> {len(all_expanded)} "
                f"(확장율: {stats['expansion_ratio']:.2f}x)"
            )

            op.tokens = sum(len(q.split()) for q in all_expanded)
            return all_expanded, stats

    async def expand_single_query(
        self, query: str, expander_func: Callable[[str], Coroutine[Any, Any, str]]
    ) -> list[str]:
        """단일 쿼리 확장 (헬퍼 함수)"""
        expanded_queries, _ = await self.expand_queries_concurrently(
            [query], expander_func
        )
        return expanded_queries


class ConcurrentDocumentRetriever:
    """
    병렬 문서 검색기 - 여러 쿼리로부터 문서를 동시에 검색

    특징:
    - 세마포어를 통한 동시 검색 제한
    - SHA256 기반 중복 제거
    - 메타데이터 통합
    - 성능 모니터링
    """

    def __init__(self, config: AsyncConfig | None = None):
        self.config = config or AsyncConfig()
        self.semaphore = AsyncSemaphore(self.config.max_concurrent_retrievals)

    async def retrieve_documents_parallel(
        self,
        queries: list[str],
        retriever_func: Callable[[str], Coroutine[Any, Any, DocumentList]],
        deduplicate: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DocumentList, dict[str, Any]]:
        """
        여러 쿼리로부터 문서를 병렬 검색

        Args:
            queries: 검색할 쿼리 리스트
            retriever_func: 쿼리를 받아서 문서 리스트를 반환하는 비동기 함수
            deduplicate: SHA256 기반 중복 제거 여부
            metadata: 성능 모니터링용 메타데이터

        Returns:
            (검색된 문서 리스트, 통계 딕셔너리)
        """
        with monitor.track_operation(
            OperationType.DOCUMENT_RETRIEVAL,
            {"query_count": len(queries), **(metadata or {})},
        ) as op:
            logger.info(
                f"[Optimizer] [Retrieval] 병렬 문서 검색 시작: {len(queries)} 쿼리"
            )

            async def _retrieve_with_limit(
                query: str, index: int
            ) -> tuple[int, DocumentList]:
                """세마포어 제한과 함께 단일 쿼리 검색"""
                async with self.semaphore:
                    try:
                        result = await asyncio.wait_for(
                            retriever_func(query), timeout=self.config.timeout_retriever
                        )
                        docs = result if isinstance(result, list) else []
                        logger.debug(
                            f"[AsyncOptimizer] 검색 {index} 완료: "
                            f"{len(docs)} 문서 (쿼리: {query[:40]})"
                        )
                        return index, docs

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[AsyncOptimizer] 검색 타임아웃 (쿼리 {index}): {query[:50]}"
                        )
                        return index, []
                    except Exception as e:
                        logger.error(f"[AsyncOptimizer] 검색 오류 (쿼리 {index}): {e}")
                        op.error = str(e)
                        return index, []

            # 모든 쿼리로부터 병렬 검색
            tasks = [_retrieve_with_limit(q, i) for i, q in enumerate(queries)]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # 결과 정렬
            sorted_results = sorted(results, key=lambda x: x[0])
            all_documents = [doc for _, docs in sorted_results for doc in docs]

            # 중복 제거
            if deduplicate and all_documents:
                unique_docs, dup_count = self._deduplicate_documents(all_documents)
            else:
                unique_docs = all_documents
                dup_count = 0

            stats = {
                "query_count": len(queries),
                "total_retrieved": len(all_documents),
                "unique_count": len(unique_docs),
                "duplicates_removed": dup_count,
                "deduplication_ratio": dup_count / len(all_documents)
                if all_documents
                else 0,
            }

            logger.info(
                f"[Optimizer] [Retrieval] 병렬 검색 완료: "
                f"{len(all_documents)} 문서 검색, "
                f"{dup_count} 중복 제거, "
                f"{len(unique_docs)} 최종 문서"
            )

            op.tokens = sum(len(doc.page_content.split()) for doc in unique_docs)
            return unique_docs, stats

    def _deduplicate_documents(
        self, documents: DocumentList
    ) -> tuple[DocumentList, int]:
        """
        고성능 튜플 기반 중복 제거 (SHA256보다 빠름)

        Args:
            documents: 원본 문서 리스트

        Returns:
            (중복 제거된 문서, 제거된 문서 수)
        """
        unique_docs = []
        seen = set()
        duplicate_count = 0

        for doc in documents:
            # 문서 내용 + 출처 정보를 함께 튜플로 구성 (Python set에서 효율적으로 해싱됨)
            doc_key = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))

            if doc_key not in seen:
                unique_docs.append(doc)
                seen.add(doc_key)
            else:
                duplicate_count += 1

        return unique_docs, duplicate_count


class ConcurrentDocumentReranker:
    """
    병렬 문서 리랭킹기 - 큰 문서 집합을 배치로 나누어 동시 리랭킹

    특징:
    - 배치 처리로 메모리 효율화
    - 세마포어를 통한 동시 배치 제한
    - 스코어 기반 정렬
    - 성능 모니터링
    """

    def __init__(self, config: AsyncConfig | None = None):
        self.config = config or AsyncConfig()
        self.semaphore = AsyncSemaphore(self.config.max_concurrent_rerankings)

    async def rerank_documents_parallel(
        self,
        query: str,
        documents: DocumentList,
        reranker_func: Callable[[str, DocumentList], Coroutine[Any, Any, list[float]]],
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DocumentList, dict[str, Any]]:
        """
        문서를 배치로 나누어 동시 리랭킹

        Args:
            query: 원본 쿼리
            documents: 리랭킹할 문서 리스트
            reranker_func: 쿼리와 문서를 받아서 스코어를 반환하는 비동기 함수
            top_k: 반환할 상위 문서 수
            metadata: 성능 모니터링용 메타데이터

        Returns:
            (정렬된 문서 리스트, 통계 딕셔너리)
        """
        with monitor.track_operation(
            OperationType.DOCUMENT_RERANKING,
            {"doc_count": len(documents), "query": query, **(metadata or {})},
        ) as op:
            logger.info(f"[Optimizer] [Rerank] 병렬 리랭킹 시작: {len(documents)} 문서")

            if not documents:
                return [], {"input_count": 0, "output_count": 0}

            # 배치로 문서 분할
            batch_size = self.config.batch_size_reranking
            batches = [
                documents[i : i + batch_size]
                for i in range(0, len(documents), batch_size)
            ]

            logger.info(
                f"[AsyncOptimizer] 리랭킹 배치: "
                f"{len(batches)} 배치 (배치 크기: {batch_size})"
            )

            async def _rerank_batch(
                batch: DocumentList, batch_idx: int
            ) -> tuple[int, list[tuple[Document, float]]]:
                """배치 리랭킹"""
                async with self.semaphore:
                    try:
                        scores = await asyncio.wait_for(
                            reranker_func(query, batch),
                            timeout=self.config.timeout_reranking,
                        )

                        scored_pairs = list(zip(batch, scores, strict=False))
                        logger.debug(
                            f"[AsyncOptimizer] 배치 {batch_idx} 리랭킹 완료: "
                            f"{len(scored_pairs)} 문서"
                        )
                        return batch_idx, scored_pairs

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[AsyncOptimizer] 리랭킹 배치 타임아웃 (배치 {batch_idx})"
                        )
                        # 폴백: 스코어 없이 반환 (뒤에서 점수 낮음)
                        return batch_idx, [(doc, 0.0) for doc in batch]
                    except Exception as e:
                        logger.error(
                            f"[AsyncOptimizer] 리랭킹 배치 오류 (배치 {batch_idx}): {e}"
                        )
                        op.error = str(e)
                        return batch_idx, [(doc, 0.0) for doc in batch]

            # 모든 배치를 동시 리랭킹
            tasks = [_rerank_batch(batch, i) for i, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)

            # 결과 수집 및 정렬
            all_scored: list[tuple[Document, float]] = []
            for _, scored_pairs in sorted(batch_results, key=lambda x: x[0]):
                all_scored.extend(scored_pairs)

            # 최종 정렬 및 상위 K개 선택
            sorted_docs = sorted(all_scored, key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in sorted_docs[:top_k]]

            stats = {
                "input_count": len(documents),
                "output_count": len(final_docs),
                "batch_count": len(batches),
                "batch_size": batch_size,
            }

            logger.info(
                f"[Optimizer] [Rerank] 병렬 리랭킹 완료: "
                f"{len(all_scored)} 결과 도출 (상위 {top_k}개 유지)"
            )

            op.tokens = sum(len(doc.page_content.split()) for doc in final_docs)
            return final_docs, stats


class ConcurrentEmbeddingGenerator:
    """
    병렬 임베딩 생성기 - 여러 텍스트를 동시에 임베딩

    특징:
    - 배치 처리로 메모리 효율화
    - 세마포어를 통한 동시 요청 제한
    - 중복 캐싱 가능 (선택)
    - 성능 모니터링
    """

    def __init__(self, config: AsyncConfig | None = None):
        self.config = config or AsyncConfig()
        self.semaphore = AsyncSemaphore(self.config.max_concurrent_embeddings)
        self.embedding_cache: dict[str, Any] = {}

    async def generate_embeddings_parallel(
        self,
        texts: list[str],
        embedding_func: Callable[[list[str]], Coroutine[Any, Any, Any]],
        use_cache: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        여러 텍스트를 배치로 병렬 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트
            embedding_func: 텍스트 리스트를 받아서 임베딩을 반환하는 비동기 함수
            use_cache: 임베딩 캐시 사용 여부
            metadata: 성능 모니터링용 메타데이터

        Returns:
            (임베딩 리스트, 통계 딕셔너리)
        """
        batch_size = self.config.batch_size_embeddings
        with monitor.track_operation(
            OperationType.EMBEDDING_GENERATION,
            {"text_count": len(texts), "batch_size": batch_size, **(metadata or {})},
        ) as op:
            logger.info(
                f"[Optimizer] [Embedding] 병렬 임베딩 생성 시작: {len(texts)} 텍스트"
            )

            if not texts:
                return [], {"input_count": 0, "cache_hits": 0}

            # 캐시된 임베딩과 새로운 텍스트 분리
            embeddings_result = [None] * len(texts)
            texts_to_embed = []
            text_indices = []
            cache_hits = 0

            if use_cache:
                for i, text in enumerate(texts):
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    if text_hash in self.embedding_cache:
                        embeddings_result[i] = self.embedding_cache[text_hash]
                        cache_hits += 1
                    else:
                        texts_to_embed.append(text)
                        text_indices.append(i)
            else:
                texts_to_embed = texts
                text_indices = list(range(len(texts)))

            logger.info(f"[AsyncOptimizer] 임베딩 캐시: {cache_hits}/{len(texts)} 히트")

            # 배치로 텍스트 분할
            batch_size = self.config.batch_size_embeddings
            batches = [
                texts_to_embed[i : i + batch_size]
                for i in range(0, len(texts_to_embed), batch_size)
            ]

            logger.info(
                f"[AsyncOptimizer] 임베딩 배치: "
                f"{len(batches)} 배치 (배치 크기: {batch_size})"
            )

            async def _embed_batch(
                batch: list[str], batch_idx: int
            ) -> tuple[int, list[Any]]:
                """배치 임베딩"""
                async with self.semaphore:
                    try:
                        embeddings = await asyncio.wait_for(
                            embedding_func(batch), timeout=self.config.timeout_embedding
                        )
                        logger.debug(
                            f"[AsyncOptimizer] 임베딩 배치 {batch_idx} 완료: "
                            f"{len(embeddings)} 임베딩"
                        )
                        return batch_idx, embeddings

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[AsyncOptimizer] 임베딩 배치 타임아웃 (배치 {batch_idx})"
                        )
                        return batch_idx, [None] * len(batch)
                    except Exception as e:
                        logger.error(
                            f"[AsyncOptimizer] 임베딩 배치 오류 (배치 {batch_idx}): {e}"
                        )
                        op.error = str(e)
                        return batch_idx, [None] * len(batch)

            # 모든 배치를 동시 임베딩
            if texts_to_embed:
                tasks = [_embed_batch(batch, i) for i, batch in enumerate(batches)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=False)

                # 결과를 원래 인덱스에 배치
                for batch_idx, embeddings in batch_results:
                    start_idx = batch_idx * batch_size
                    for local_idx, embedding in enumerate(embeddings):
                        original_idx = text_indices[start_idx + local_idx]
                        embeddings_result[original_idx] = embedding

                        # 캐시에 저장
                        if use_cache and embedding is not None:
                            text_hash = hashlib.sha256(
                                texts[original_idx].encode()
                            ).hexdigest()
                            self.embedding_cache[text_hash] = embedding

            stats = {
                "input_count": len(texts),
                "cache_hits": cache_hits,
                "cache_misses": len(texts_to_embed),
                "batch_count": len(batches),
                "batch_size": batch_size,
            }

            logger.info(
                f"[Optimizer] [Embedding] 병렬 임베딩 완료: {len(embeddings)} 벡터 생성"
            )

            op.tokens = sum(len(text.split()) for text in texts)
            return embeddings_result, stats

    def clear_cache(self) -> dict[str, Any]:
        """임베딩 캐시 초기화"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"[AsyncOptimizer] 임베딩 캐시 초기화: {cache_size} 항목")
        return {"cleared_entries": cache_size}


# 전역 인스턴스
_async_config: AsyncConfig | None = None


def get_async_config() -> AsyncConfig:
    """전역 AsyncConfig 인스턴스 반환"""
    global _async_config
    if _async_config is None:
        _async_config = AsyncConfig()
    return _async_config


def set_async_config(config: AsyncConfig) -> None:
    """전역 AsyncConfig 인스턴스 설정"""
    global _async_config
    _async_config = config
    logger.info(f"[AsyncOptimizer] AsyncConfig 설정: {config}")


def get_concurrent_query_expander(
    config: AsyncConfig | None = None,
) -> ConcurrentQueryExpander:
    """ConcurrentQueryExpander 인스턴스 반환"""
    return ConcurrentQueryExpander(config or get_async_config())


def get_concurrent_document_retriever(
    config: AsyncConfig | None = None,
) -> ConcurrentDocumentRetriever:
    """ConcurrentDocumentRetriever 인스턴스 반환"""
    return ConcurrentDocumentRetriever(config or get_async_config())


def get_concurrent_document_reranker(
    config: AsyncConfig | None = None,
) -> ConcurrentDocumentReranker:
    """ConcurrentDocumentReranker 인스턴스 반환"""
    return ConcurrentDocumentReranker(config or get_async_config())


def get_concurrent_embedding_generator(
    config: AsyncConfig | None = None,
) -> ConcurrentEmbeddingGenerator:
    """ConcurrentEmbeddingGenerator 인스턴스 반환"""
    return ConcurrentEmbeddingGenerator(config or get_async_config())
