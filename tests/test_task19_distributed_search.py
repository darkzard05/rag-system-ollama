"""
Task 19 테스트 스위트: 분산 검색 및 재순위지정
"""

import pytest
from src.services.distributed.distributed_search import (
    DistributedSearchExecutor,
    SearchQuery,
    SearchResult,
    NodeSearchStatus,
)
from src.core.search_aggregator import (
    SearchResultAggregator,
    AggregationStrategy,
    DuplicateStrategy,
)
from src.core.reranker import (
    DistributedReranker,
    RerankerStrategy,
    RerankingResult,
    SimilarityCalculator,
)


@pytest.fixture
def executor():
    """분산 검색 실행기"""
    return DistributedSearchExecutor(num_nodes=3, docs_per_node=50)


@pytest.fixture
def aggregator():
    """검색 결과 집계기"""
    return SearchResultAggregator(dedup_strategy=DuplicateStrategy.KEEP_HIGHEST_SCORE)


@pytest.fixture
def reranker():
    """재순위지정 엔진"""
    return DistributedReranker()


@pytest.fixture
def query_embedding():
    """쿼리 임베딩"""
    return [float(i % 10) / 10 for i in range(384)]


# ==============================================
# 테스트 그룹 1: 분산 검색 (5개 테스트)
# ==============================================


class TestDistributedSearch:
    """분산 검색 테스트"""

    def test_01_query_submission(self, executor, query_embedding):
        """쿼리 제출 및 ID 생성"""
        query_id = executor.submit_query("test query", query_embedding, top_k=10)

        assert query_id is not None
        assert len(query_id) > 0
        assert query_id in executor._active_searches

        query = executor._active_searches[query_id]
        assert query.query_text == "test query"
        assert query.top_k == 10

    def test_02_single_node_search(self, executor, query_embedding):
        """단일 노드 검색"""
        query = SearchQuery(
            query_id="test", query_text="test query", embedding=query_embedding, top_k=5
        )

        task = executor.execute_search_on_node("node_0", query)

        assert task.status == NodeSearchStatus.COMPLETED
        assert len(task.results) > 0
        assert len(task.results) <= 5
        assert task.execution_time > 0

    def test_03_distributed_search(self, executor, query_embedding):
        """분산 검색 (모든 노드)"""
        query_id = executor.submit_query("distributed test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        assert len(results) > 0
        # 3개 노드 × 10 top_k = 최대 30개
        assert len(results) <= 30

        # 모든 결과는 SearchResult 인스턴스
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.node_id is not None

    def test_04_parallel_search(self, executor, query_embedding):
        """병렬 검색 (쓰레드 기반)"""
        query_id = executor.submit_query("parallel test", query_embedding, top_k=15)
        results = executor.execute_parallel_search(query_id, num_threads=3)

        assert len(results) > 0
        # 3개 노드에서 병렬로 검색
        assert all(isinstance(r, SearchResult) for r in results)

    def test_05_search_with_filters(self, executor, query_embedding):
        """필터를 포함한 검색"""
        query = SearchQuery(
            query_id="filtered",
            query_text="filtered query",
            embedding=query_embedding,
            top_k=5,
            filters={"source": "source_0"},
        )

        task = executor.execute_search_on_node("node_0", query)
        results = task.results

        # 필터 적용 여부 확인
        for result in results:
            if result.metadata:
                assert result.metadata.get("source") == "source_0"


# ==============================================
# 테스트 그룹 2: 결과 집계 (5개 테스트)
# ==============================================


class TestSearchAggregation:
    """검색 결과 집계 테스트"""

    def test_06_merge_all_strategy(self, executor, aggregator, query_embedding):
        """MERGE_ALL 전략"""
        # 검색 실행
        query_id = executor.submit_query("merge test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        # 노드별 결과 분류
        search_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in results:
            search_results[result.node_id].append(result)

        # 집계
        aggregated, metrics = aggregator.aggregate_results(
            search_results, AggregationStrategy.MERGE_ALL, top_k=15
        )

        assert len(aggregated) > 0
        assert metrics.total_output_results <= 15
        assert all(hasattr(r, "doc_id") for r in aggregated)

    def test_07_dedup_by_content(self, executor, aggregator, query_embedding):
        """DEDUP_CONTENT 전략"""
        query_id = executor.submit_query("dedup test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        search_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in results:
            search_results[result.node_id].append(result)

        aggregated, metrics = aggregator.aggregate_results(
            search_results, AggregationStrategy.DEDUP_CONTENT
        )

        # 중복 제거되므로 결과가 감소
        assert metrics.duplicates_found >= 0
        assert all(hasattr(r, "occurrence_count") for r in aggregated)

    def test_08_dedup_by_id(self, executor, aggregator, query_embedding):
        """DEDUP_ID 전략"""
        query_id = executor.submit_query("id dedup test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        search_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in results:
            search_results[result.node_id].append(result)

        aggregated, metrics = aggregator.aggregate_results(
            search_results, AggregationStrategy.DEDUP_ID
        )

        # ID 중복 제거 확인
        doc_ids = [r.doc_id for r in aggregated]
        assert len(doc_ids) == len(set(doc_ids))  # 중복 없음

    def test_09_weighted_score(self, executor, aggregator, query_embedding):
        """WEIGHTED_SCORE 전략"""
        query_id = executor.submit_query("weighted test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        search_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in results:
            search_results[result.node_id].append(result)

        aggregated, metrics = aggregator.aggregate_results(
            search_results, AggregationStrategy.WEIGHTED_SCORE
        )

        # 가중 점수 확인
        for result in aggregated:
            assert 0.0 <= result.aggregated_score <= 1.0
            assert (
                result.score_adjustments >= 0
                if hasattr(result, "score_adjustments")
                else True
            )

    def test_10_aggregation_metrics(self, executor, aggregator, query_embedding):
        """집계 메트릭"""
        query_id = executor.submit_query("metrics test", query_embedding, top_k=10)
        results = executor.execute_distributed_search(query_id)

        search_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in results:
            search_results[result.node_id].append(result)

        aggregated, metrics = aggregator.aggregate_results(
            search_results, AggregationStrategy.MERGE_ALL
        )

        # 메트릭 검증
        assert metrics.total_input_results > 0
        assert metrics.total_output_results > 0
        assert metrics.aggregation_time >= 0.0
        assert metrics.nodes_processed == 3


# ==============================================
# 테스트 그룹 3: 재순위지정 (6개 테스트)
# ==============================================


class TestReranking:
    """재순위지정 테스트"""

    def test_11_rerank_by_score(self, executor, reranker, query_embedding):
        """점수 기반 재순위지정"""
        query_id = executor.submit_query("rerank test", query_embedding, top_k=20)
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, strategy=RerankerStrategy.SCORE_ONLY
        )

        assert len(reranked) > 0
        assert all(isinstance(r, RerankingResult) for r in reranked)

        # 점수 기반 정렬 확인
        for i in range(len(reranked) - 1):
            assert reranked[i].reranked_score >= reranked[i + 1].reranked_score

    def test_12_rerank_by_diversity(self, executor, reranker, query_embedding):
        """다양성 기반 재순위지정"""
        query_id = executor.submit_query("diversity test", query_embedding, top_k=20)
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, strategy=RerankerStrategy.DIVERSITY, diversity_weight=0.4
        )

        assert len(reranked) > 0
        assert metrics.rank_changes >= 0
        assert metrics.max_rank_change >= 0

    def test_13_rerank_by_cross_encoder(self, executor, reranker, query_embedding):
        """교차 인코더 재순위지정"""
        query_id = executor.submit_query(
            "cross encoder test", query_embedding, top_k=20
        )
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, "search query", strategy=RerankerStrategy.CROSS_ENCODER
        )

        assert len(reranked) > 0
        # 모든 결과에 관련성 점수 포함
        for r in reranked:
            assert "relevance" in r.reranking_factors

    def test_14_rerank_by_mmr(self, executor, reranker, query_embedding):
        """MMR 재순위지정"""
        query_id = executor.submit_query("mmr test", query_embedding, top_k=20)
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, strategy=RerankerStrategy.MMR, mmr_lambda=0.5
        )

        assert len(reranked) > 0
        for r in reranked:
            assert "mmr" in r.reranking_factors

    def test_15_rerank_by_fusion(self, executor, reranker, query_embedding):
        """통합 재순위지정"""
        query_id = executor.submit_query("fusion test", query_embedding, top_k=20)
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, "search query", strategy=RerankerStrategy.FUSION
        )

        assert len(reranked) > 0
        for r in reranked:
            assert "fusion_rank" in r.reranking_factors

    def test_16_reranking_metrics(self, executor, reranker, query_embedding):
        """재순위지정 메트릭"""
        query_id = executor.submit_query("metrics test", query_embedding, top_k=20)
        results = executor.execute_distributed_search(query_id)

        reranked, metrics = reranker.rerank(
            results, strategy=RerankerStrategy.DIVERSITY
        )

        # 메트릭 검증
        assert metrics.total_results > 0
        assert metrics.reranked_results > 0
        assert metrics.reranking_time > 0.0
        assert metrics.avg_score_before >= 0.0
        assert metrics.avg_score_after >= 0.0


# ==============================================
# 테스트 그룹 4: 유사도 계산 (2개 테스트)
# ==============================================


class TestSimilarityCalculation:
    """유사도 계산 테스트"""

    def test_17_cosine_similarity(self):
        """코사인 유사도"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01  # 완전히 같음

        vec3 = [0.0, 1.0, 0.0]
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec3)
        assert abs(similarity) < 0.01  # 수직

    def test_18_euclidean_distance(self):
        """유클리디안 거리"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        distance = SimilarityCalculator.euclidean_distance(vec1, vec2)
        assert abs(distance - 1.0) < 0.01


# ==============================================
# 테스트 그룹 5: 엔드-투-엔드 (2개 테스트)
# ==============================================


class TestEndToEnd:
    """엔드-투-엔드 테스트"""

    def test_19_complete_search_pipeline(
        self, executor, aggregator, reranker, query_embedding
    ):
        """완전한 검색 파이프라인"""
        # 1단계: 분산 검색
        query_id = executor.submit_query("pipeline test", query_embedding, top_k=20)
        search_results = executor.execute_distributed_search(query_id)

        # 2단계: 결과 집계
        node_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in search_results:
            node_results[result.node_id].append(result)

        aggregated, agg_metrics = aggregator.aggregate_results(
            node_results, AggregationStrategy.DEDUP_CONTENT
        )

        # 3단계: 재순위지정
        reranked, rerank_metrics = reranker.rerank(
            aggregated, "pipeline query", strategy=RerankerStrategy.FUSION
        )

        # 검증
        assert len(reranked) > 0
        assert all(isinstance(r, RerankingResult) for r in reranked)
        assert agg_metrics.aggregation_time > 0
        assert rerank_metrics.reranking_time > 0

    def test_20_search_with_all_strategies(
        self, executor, aggregator, reranker, query_embedding
    ):
        """모든 전략 테스트"""
        query_id = executor.submit_query("all strategies", query_embedding, top_k=15)
        results = executor.execute_distributed_search(query_id)

        # 모든 재순위지정 전략 테스트
        strategies = [
            RerankerStrategy.SCORE_ONLY,
            RerankerStrategy.DIVERSITY,
            RerankerStrategy.RECENCY,
            RerankerStrategy.CROSS_ENCODER,
            RerankerStrategy.MMR,
            RerankerStrategy.FUSION,
        ]

        for strategy in strategies:
            reranked, metrics = reranker.rerank(
                results, "test query", strategy=strategy
            )

            assert len(reranked) > 0
            assert metrics.strategy_used == strategy.value

            # 정렬 검증
            for i in range(len(reranked) - 1):
                assert reranked[i].reranked_score >= reranked[i + 1].reranked_score


# ==============================================
# 테스트 그룹 6: 통합 검증 (1개 테스트)
# ==============================================


class TestIntegrationValidation:
    """통합 검증 테스트"""

    def test_21_full_system_integration(
        self, executor, aggregator, reranker, query_embedding
    ):
        """전체 시스템 통합 검증"""
        # 여러 쿼리 실행
        query_ids = []
        for i in range(3):
            qid = executor.submit_query(
                f"integration test {i}", query_embedding, top_k=10 + i * 5
            )
            query_ids.append(qid)

        # 모든 쿼리 검색
        for query_id in query_ids:
            results = executor.execute_distributed_search(query_id)
            assert len(results) > 0

        # 최종 검색 - 전체 파이프라인
        final_query_id = executor.submit_query("final test", query_embedding, top_k=25)
        final_results = executor.execute_parallel_search(final_query_id)

        # 집계
        node_results = {"node_0": [], "node_1": [], "node_2": []}
        for result in final_results:
            node_results[result.node_id].append(result)

        aggregated, _ = aggregator.aggregate_results(
            node_results, AggregationStrategy.WEIGHTED_SCORE, top_k=20
        )

        # 재순위지정
        final_reranked, final_metrics = reranker.rerank(
            aggregated, "final query", strategy=RerankerStrategy.FUSION, top_k=10
        )

        # 최종 검증
        assert len(final_reranked) <= 10
        assert all(r.final_rank < 10 for r in final_reranked)
        # 시간은 0일 수도 있으므로 >= 검사
        assert final_metrics.reranking_time >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
