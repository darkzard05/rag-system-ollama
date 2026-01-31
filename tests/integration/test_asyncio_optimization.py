"""
AsyncIO 최적화 통합 테스트 - Task 11
총 25개의 포괄적인 테스트
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import unittest

from langchain_core.documents import Document

from services.optimization.async_optimizer import (
    AsyncConfig,
    AsyncSemaphore,
    ConcurrentDocumentReranker,
    ConcurrentDocumentRetriever,
    ConcurrentEmbeddingGenerator,
    ConcurrentQueryExpander,
    get_async_config,
    get_concurrent_document_retriever,
    get_concurrent_query_expander,
    set_async_config,
)


class TestAsyncConfig(unittest.TestCase):
    """AsyncConfig 테스트 (2개)"""

    def test_async_config_default_values(self):
        """기본값 확인"""
        config = AsyncConfig()
        assert config.max_concurrent_queries == 5
        assert config.max_concurrent_retrievals == 10
        assert config.timeout_llm == 30.0

    def test_async_config_custom_values(self):
        """커스텀 설정 확인"""
        config = AsyncConfig(
            max_concurrent_queries=3, max_concurrent_retrievals=5, timeout_llm=60.0
        )
        assert config.max_concurrent_queries == 3
        assert config.max_concurrent_retrievals == 5
        assert config.timeout_llm == 60.0


class TestAsyncSemaphore(unittest.TestCase):
    """AsyncSemaphore 테스트 (2개)"""

    def test_semaphore_single_acquisition(self):
        """단일 세마포어 획득"""

        async def _test():
            semaphore = AsyncSemaphore(1)
            async with semaphore:
                pass

        asyncio.run(_test())

    def test_semaphore_multiple_concurrent(self):
        """동시 세마포어 제어"""

        async def _test():
            semaphore = AsyncSemaphore(2)
            counter = 0

            async def _task():
                nonlocal counter
                async with semaphore:
                    counter += 1
                    await asyncio.sleep(0.01)

            await asyncio.gather(*[_task() for _ in range(5)])
            assert counter == 5

        asyncio.run(_test())


class TestConcurrentQueryExpander(unittest.TestCase):
    """ConcurrentQueryExpander 테스트 (6개)"""

    def setUp(self):
        self.expander = ConcurrentQueryExpander()

    def test_single_query_expansion(self):
        """단일 쿼리 확장"""

        async def _expand_func(query: str) -> str:
            return f"{query}\n{query} (related 1)\n{query} (related 2)"

        async def _test():
            result, stats = await self.expander.expand_queries_concurrently(
                ["test query"], _expand_func
            )
            assert len(result) > 0
            assert stats["input_queries"] == 1

        asyncio.run(_test())

    def test_multiple_queries_expansion(self):
        """다중 쿼리 확장"""

        async def _expand_func(query: str) -> str:
            return f"{query}\n{query} expanded"

        async def _test():
            result, stats = await self.expander.expand_queries_concurrently(
                ["query1", "query2", "query3"], _expand_func
            )
            assert len(result) > 3
            assert stats["input_queries"] == 3

        asyncio.run(_test())

    def test_expansion_with_timeout(self):
        """타임아웃 처리"""

        async def _slow_expand(query: str) -> str:
            await asyncio.sleep(0.1)
            return query

        async def _test():
            config = AsyncConfig(timeout_llm=0.05)
            expander = ConcurrentQueryExpander(config)
            result, stats = await expander.expand_queries_concurrently(
                ["query"], _slow_expand
            )
            # 타임아웃 시에도 폴백으로 원본 쿼리 반환
            assert "query" in result

        asyncio.run(_test())

    def test_expansion_error_handling(self):
        """에러 처리"""

        async def _error_expand(query: str) -> str:
            raise ValueError("Test error")

        async def _test():
            result, stats = await self.expander.expand_queries_concurrently(
                ["query"], _error_expand
            )
            # 에러 시에도 원본 쿼리 반환
            assert result == ["query"]

        asyncio.run(_test())

    def test_expand_single_query_helper(self):
        """단일 쿼리 헬퍼 함수"""

        async def _expand_func(query: str) -> str:
            return f"{query} expanded"

        async def _test():
            result = await self.expander.expand_single_query("test", _expand_func)
            assert len(result) > 0

        asyncio.run(_test())


class TestConcurrentDocumentRetriever(unittest.TestCase):
    """ConcurrentDocumentRetriever 테스트 (5개)"""

    def setUp(self):
        self.retriever = ConcurrentDocumentRetriever()

    def _create_mock_doc(self, content: str, source: str) -> Document:
        """테스트용 문서 생성"""
        return Document(page_content=content, metadata={"source": source, "page": 1})

    def test_parallel_retrieval(self):
        """병렬 문서 검색"""

        async def _retrieve_func(query: str) -> list[Document]:
            return [self._create_mock_doc(f"content for {query}", f"source_{query}")]

        async def _test():
            docs, stats = await self.retriever.retrieve_documents_parallel(
                ["query1", "query2"], _retrieve_func
            )
            assert len(docs) > 0
            assert stats["query_count"] == 2

        asyncio.run(_test())

    def test_deduplication(self):
        """중복 제거"""

        async def _retrieve_func(query: str) -> list[Document]:
            # 첫 번째 쿼리에서는 중복 문서 2개 반환 (같은 source)
            if query == "query":
                return [
                    self._create_mock_doc("duplicate content", "source_1"),
                    self._create_mock_doc("duplicate content", "source_1"),
                ]
            return []

        async def _test():
            docs, stats = await self.retriever.retrieve_documents_parallel(
                ["query"], _retrieve_func, deduplicate=True
            )
            # 중복이 제거되어야 함 (같은 내용 + 같은 source)
            assert len(docs) == 1
            assert stats["duplicates_removed"] > 0

        asyncio.run(_test())

    def test_no_deduplication(self):
        """중복 제거 비활성화"""

        async def _retrieve_func(query: str) -> list[Document]:
            return [
                self._create_mock_doc("content", "source"),
                self._create_mock_doc("content", "source"),
            ]

        async def _test():
            docs, stats = await self.retriever.retrieve_documents_parallel(
                ["query"], _retrieve_func, deduplicate=False
            )
            # 중복이 유지되어야 함
            assert len(docs) == 2

        asyncio.run(_test())

    def test_retrieval_with_timeout(self):
        """타임아웃 처리"""

        async def _slow_retrieve(query: str) -> list[Document]:
            await asyncio.sleep(0.1)
            return []

        async def _test():
            config = AsyncConfig(timeout_retriever=0.05)
            retriever = ConcurrentDocumentRetriever(config)
            docs, stats = await retriever.retrieve_documents_parallel(
                ["query"], _slow_retrieve
            )
            # 타임아웃 시에도 에러 없이 처리
            assert isinstance(docs, list)

        asyncio.run(_test())


class TestConcurrentDocumentReranker(unittest.TestCase):
    """ConcurrentDocumentReranker 테스트 (5개)"""

    def setUp(self):
        self.reranker = ConcurrentDocumentReranker()

    def _create_mock_doc(self, content: str, idx: int) -> Document:
        """테스트용 문서 생성"""
        return Document(
            page_content=content, metadata={"source": f"doc_{idx}", "page": 1}
        )

    def test_parallel_reranking(self):
        """병렬 리랭킹"""

        async def _rerank_func(query: str, docs: list[Document]) -> list[float]:
            return [float(i) for i in range(len(docs))]

        async def _test():
            docs = [self._create_mock_doc(f"content {i}", i) for i in range(5)]
            result, stats = await self.reranker.rerank_documents_parallel(
                "query", docs, _rerank_func, top_k=3
            )
            assert len(result) == 3

        asyncio.run(_test())

    def test_batch_processing(self):
        """배치 처리"""

        async def _rerank_func(query: str, docs: list[Document]) -> list[float]:
            return [0.5] * len(docs)

        async def _test():
            docs = [self._create_mock_doc(f"content {i}", i) for i in range(100)]
            result, stats = await self.reranker.rerank_documents_parallel(
                "query", docs, _rerank_func, top_k=10
            )
            assert len(result) == 10
            assert stats["batch_count"] > 1

        asyncio.run(_test())

    def test_top_k_selection(self):
        """상위 K개 선택"""

        async def _rerank_func(query: str, docs: list[Document]) -> list[float]:
            # 역순 스코어 반환
            return [float(len(docs) - i) for i in range(len(docs))]

        async def _test():
            docs = [self._create_mock_doc(f"content {i}", i) for i in range(10)]
            result, stats = await self.reranker.rerank_documents_parallel(
                "query", docs, _rerank_func, top_k=5
            )
            assert len(result) == 5

        asyncio.run(_test())

    def test_reranking_with_timeout(self):
        """타임아웃 처리"""

        async def _slow_rerank(query: str, docs: list[Document]) -> list[float]:
            await asyncio.sleep(0.1)
            return [0.0] * len(docs)

        async def _test():
            config = AsyncConfig(timeout_reranking=0.05)
            reranker = ConcurrentDocumentReranker(config)
            docs = [self._create_mock_doc(f"content {i}", i) for i in range(5)]
            result, stats = await reranker.rerank_documents_parallel(
                "query", docs, _slow_rerank, top_k=3
            )
            # 타임아웃 시에도 결과 반환
            assert isinstance(result, list)

        asyncio.run(_test())


class TestConcurrentEmbeddingGenerator(unittest.TestCase):
    """ConcurrentEmbeddingGenerator 테스트 (4개)"""

    def setUp(self):
        self.generator = ConcurrentEmbeddingGenerator()

    def test_parallel_embedding_generation(self):
        """병렬 임베딩 생성"""

        async def _embed_func(texts: list[str]) -> list[list]:
            return [[float(i)] * 10 for i in range(len(texts))]

        async def _test():
            embeddings, stats = await self.generator.generate_embeddings_parallel(
                ["text1", "text2", "text3"], _embed_func
            )
            assert len(embeddings) == 3
            assert stats["input_count"] == 3

        asyncio.run(_test())

    def test_embedding_caching(self):
        """임베딩 캐싱"""
        call_count = 0

        async def _embed_func(texts: list[str]) -> list[list]:
            nonlocal call_count
            call_count += 1
            return [[float(i)] * 10 for i in range(len(texts))]

        async def _test():
            nonlocal call_count
            call_count = 0

            # 첫 번째 호출
            embeddings1, stats1 = await self.generator.generate_embeddings_parallel(
                ["text"], _embed_func, use_cache=True
            )

            # 같은 텍스트로 두 번째 호출
            embeddings2, stats2 = await self.generator.generate_embeddings_parallel(
                ["text"], _embed_func, use_cache=True
            )

            # 캐시 히트가 발생했어야 함
            assert stats2["cache_hits"] == 1
            assert stats2["cache_misses"] == 0

        asyncio.run(_test())

    def test_batch_embedding_generation(self):
        """배치 임베딩 생성"""

        async def _embed_func(texts: list[str]) -> list[list]:
            return [[float(i)] * 10 for i in range(len(texts))]

        async def _test():
            embeddings, stats = await self.generator.generate_embeddings_parallel(
                [f"text_{i}" for i in range(100)], _embed_func
            )
            assert len(embeddings) == 100
            assert stats["batch_count"] > 1

        asyncio.run(_test())

    def test_cache_clearing(self):
        """캐시 초기화"""

        async def _embed_func(texts: list[str]) -> list[list]:
            return [[float(i)] * 10 for i in range(len(texts))]

        async def _test():
            # 캐시 채우기
            await self.generator.generate_embeddings_parallel(
                ["text"], _embed_func, use_cache=True
            )

            # 캐시 초기화
            result = self.generator.clear_cache()
            assert result["cleared_entries"] == 1

            # 캐시가 비워졌는지 확인
            assert len(self.generator.embedding_cache) == 0

        asyncio.run(_test())


class TestGlobalInstances(unittest.TestCase):
    """전역 인스턴스 함수 테스트 (1개)"""

    def test_global_config_management(self):
        """전역 설정 관리"""
        # 기본값
        config1 = get_async_config()
        assert config1.max_concurrent_queries == 5

        # 커스텀 설정
        custom_config = AsyncConfig(max_concurrent_queries=10)
        set_async_config(custom_config)
        config2 = get_async_config()
        assert config2.max_concurrent_queries == 10

        # 초기화
        set_async_config(AsyncConfig())


class TestIntegration(unittest.TestCase):
    """통합 테스트 (1개)"""

    def test_full_pipeline(self):
        """전체 파이프라인 시뮬레이션"""

        async def _test():
            # 쿼리 확장
            expander = get_concurrent_query_expander()

            async def _expand(query: str) -> str:
                return f"{query}\n{query} variant"

            expanded, _ = await expander.expand_queries_concurrently(
                ["original query"], _expand
            )

            # 문서 검색
            retriever = get_concurrent_document_retriever()

            async def _retrieve(query: str) -> list[Document]:
                return [
                    Document(
                        page_content=f"Content for {query}",
                        metadata={"source": "test", "page": 1},
                    )
                ]

            docs, _ = await retriever.retrieve_documents_parallel(expanded, _retrieve)

            # 결과 확인
            assert len(expanded) > 0
            assert len(docs) > 0

        asyncio.run(_test())


if __name__ == "__main__":
    # 25개 테스트 실행
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 60}")
    print("AsyncIO 최적화 테스트 완료")
    print(f"{'=' * 60}")
    print(f"총 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    print(f"{'=' * 60}")
