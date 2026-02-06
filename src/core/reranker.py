"""
Task 19-3: Distributed Reranking Module
분산 재순위지정 - 다중 노드 결과의 최종 순위 결정
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

import numpy as np


class RerankerStrategy(Enum):
    """재순위지정 전략"""

    SCORE_ONLY = "score_only"  # 점수 기반
    DIVERSITY = "diversity"  # 다양성 고려
    RECENCY = "recency"  # 시간성 고려
    CROSS_ENCODER = "cross_encoder"  # 교차 인코더
    MMR = "mmr"  # Maximal Marginal Relevance
    FUSION = "fusion"  # 여러 전략 통합


class ScoringMethod(Enum):
    """점수 계산 방식"""

    LINEAR = "linear"
    SIGMOID = "sigmoid"
    RANK_BASED = "rank_based"
    PERCENTILE = "percentile"


@dataclass
class RerankingResult:
    """재순위지정된 결과"""

    doc_id: str
    content: str
    original_score: float
    reranked_score: float
    original_rank: int
    final_rank: int
    reranking_factors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        """점수 기반 비교"""
        return self.reranked_score > other.reranked_score


@dataclass
class RerankingMetrics:
    """재순위지정 메트릭"""

    total_results: int = 0
    reranked_results: int = 0
    rank_changes: int = 0
    reranking_time: float = 0.0
    strategy_used: str = ""
    avg_score_before: float = 0.0
    avg_score_after: float = 0.0
    diversity_score: float = 0.0
    max_rank_change: int = 0


class SimilarityCalculator:
    """유사도 계산기 (Numpy 기반 최적화)"""

    @staticmethod
    def cosine_similarity(
        vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray
    ) -> float:
        """코사인 유사도"""
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def manhattan_distance(
        vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray
    ) -> float:
        """맨해튼 거리"""
        return float(np.sum(np.abs(np.array(vec1) - np.array(vec2))))

    @staticmethod
    def euclidean_distance(
        vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray
    ) -> float:
        """유클리디안 거리"""
        return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))


class DiversityCalculator:
    """다양성 계산기 (NumPy 비트셋 최적화)"""

    def __init__(self):
        self._lock = RLock()
        # [최적화] 비트셋 캐시 (256비트 uint8 배열)
        self._bitset_cache: dict[str, np.ndarray] = {}

    def _get_bitset(self, doc_id: str, content: str) -> np.ndarray:
        """텍스트의 문자 특징을 256비트 비트셋으로 변환"""
        if doc_id not in self._bitset_cache:
            # ASCII/UTF-8 하위 256자에 대한 출현 여부를 비트로 저장
            bitset = np.zeros(256, dtype=np.bool_)
            chars = np.frombuffer(
                content.encode("utf-8", errors="ignore"), dtype=np.uint8
            )
            if len(chars) > 0:
                bitset[chars] = True
            self._bitset_cache[doc_id] = bitset
        return self._bitset_cache[doc_id]

    def calculate_diversity_penalty(
        self, result: Any, selected_results: list[Any], diversity_weight: float = 0.3
    ) -> float:
        """
        다양성 페널티 계산 (비트셋 연산 적용)
        """
        if not selected_results:
            return 0.0

        # 메타데이터 기반 유사도
        metadata_similarity = self._calculate_metadata_similarity(
            result, selected_results
        )

        # 콘텐츠 기반 유사도 (비트셋 최적화)
        content_similarity = self._calculate_content_similarity_bitset(
            result, selected_results
        )

        # 통합 유사도
        avg_similarity = (metadata_similarity + content_similarity) / 2
        penalty = avg_similarity * diversity_weight

        return penalty

    def _calculate_metadata_similarity(
        self, result: Any, selected_results: list[Any]
    ) -> float:
        """메타데이터 유사도 최적화"""
        res_meta = getattr(result, "metadata", {}) or {}
        if not res_meta:
            return 0.0

        similarities = []
        for selected in selected_results:
            sel_meta = getattr(selected, "metadata", {}) or {}
            if not sel_meta:
                similarities.append(0.0)
                continue

            match_count = 0
            # 공통 키에 대해서만 비교
            common_keys = res_meta.keys() & sel_meta.keys()
            for key in common_keys:
                if res_meta[key] == sel_meta[key]:
                    match_count += 1

            total_keys = max(len(res_meta), len(sel_meta))
            similarities.append(match_count / total_keys if total_keys > 0 else 0.0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_content_similarity_bitset(
        self, result: Any, selected_results: list[Any]
    ) -> float:
        """콘텐츠 유사도 (비트셋 행렬 연산 최적화)"""
        if not selected_results:
            return 0.0

        res_id = getattr(result, "doc_id", str(hash(result.content)))
        res_bitset = self._get_bitset(res_id, result.content)

        # [최적화] 선택된 결과들의 비트셋을 행렬로 구성하여 한 번에 연산
        selected_bitsets = []
        for selected in selected_results:
            sel_id = getattr(selected, "doc_id", str(hash(selected.content)))
            selected_bitsets.append(self._get_bitset(sel_id, selected.content))

        selected_matrix = np.array(selected_bitsets)

        # 비트셋 유사도 (Jaccard Index) 행렬 연산
        intersections = np.logical_and(res_bitset, selected_matrix).sum(axis=1)
        unions = np.logical_or(res_bitset, selected_matrix).sum(axis=1)

        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions != 0,
        )

        return float(np.mean(similarities))


class QueryRelevanceScorer:
    """쿼리 관련성 점수 계산기"""

    def __init__(self):
        self._lock = RLock()

    def calculate_relevance_score(
        self,
        result: Any,
        query_text: str,
        scoring_method: ScoringMethod = ScoringMethod.LINEAR,
    ) -> float:
        """
        쿼리 관련성 점수 계산

        Args:
            result: 검색 결과
            query_text: 쿼리 텍스트
            scoring_method: 점수 계산 방식

        Returns:
            관련성 점수 (0~1)
        """
        if not query_text:
            return 0.5

        # 기본 유사도 (단어 매칭)
        query_words = set(query_text.lower().split())
        content_words = set(result.content.lower().split())

        if not query_words or not content_words:
            return 0.0

        match_ratio = len(query_words & content_words) / len(
            query_words | content_words
        )

        if scoring_method == ScoringMethod.LINEAR:
            return match_ratio

        elif scoring_method == ScoringMethod.SIGMOID:
            # 시그모이드 곡선
            return 1.0 / (1.0 + math.exp(-10 * (match_ratio - 0.5)))

        elif scoring_method == ScoringMethod.RANK_BASED:
            # 순위 기반 (현재 점수를 순위로 변환)
            return min(1.0, match_ratio * 1.5)

        elif scoring_method == ScoringMethod.PERCENTILE:
            # 백분위수 기반
            return match_ratio**0.5

        else:
            return match_ratio


class DistributedReranker:
    """분산 재순위지정 엔진"""

    def __init__(self):
        self._lock = RLock()
        self._diversity_calc = DiversityCalculator()
        self._relevance_scorer = QueryRelevanceScorer()

    def rerank(
        self,
        results: list[Any],
        query_text: str | None = None,
        strategy: RerankerStrategy = RerankerStrategy.SCORE_ONLY,
        top_k: int | None = None,
        **kwargs,
    ) -> tuple[list[RerankingResult], RerankingMetrics]:
        """
        검색 결과 재순위지정

        Args:
            results: 원본 검색 결과
            query_text: 쿼리 텍스트 (일부 전략에서 필요)
            strategy: 재순위지정 전략
            top_k: 상위 k개 결과
            **kwargs: 전략별 추가 파라미터

        Returns:
            (재순위지정된 결과, 메트릭)
        """
        start_time = time.time()
        metrics = RerankingMetrics(
            total_results=len(results), strategy_used=strategy.value
        )

        # 원본 점수 통계
        if results:
            if hasattr(results[0], "aggregated_score"):
                metrics.avg_score_before = sum(
                    r.aggregated_score for r in results
                ) / len(results)
            else:
                metrics.avg_score_before = sum(r.score for r in results) / len(results)

        # 전략별 재순위지정
        if strategy == RerankerStrategy.SCORE_ONLY:
            reranked = self._rerank_by_score(results, metrics)

        elif strategy == RerankerStrategy.DIVERSITY:
            reranked = self._rerank_by_diversity(results, metrics, **kwargs)

        elif strategy == RerankerStrategy.RECENCY:
            reranked = self._rerank_by_recency(results, metrics)

        elif strategy == RerankerStrategy.CROSS_ENCODER:
            reranked = self._rerank_by_cross_encoder(
                results, query_text, metrics, **kwargs
            )

        elif strategy == RerankerStrategy.MMR:
            reranked = self._rerank_by_mmr(results, metrics, **kwargs)

        elif strategy == RerankerStrategy.FUSION:
            reranked = self._rerank_by_fusion(results, query_text, metrics, **kwargs)

        else:
            reranked = self._rerank_by_score(results, metrics)

        # top-k 적용
        if top_k:
            reranked = reranked[:top_k]

        # 메트릭 업데이트
        metrics.reranked_results = len(reranked)
        if reranked:
            metrics.avg_score_after = sum(r.reranked_score for r in reranked) / len(
                reranked
            )
        metrics.reranking_time = time.time() - start_time

        return reranked, metrics

    def _rerank_by_score(
        self, results: list[Any], metrics: RerankingMetrics
    ) -> list[RerankingResult]:
        """점수만 기반 재순위지정"""
        reranked = []

        for idx, result in enumerate(results):
            score = (
                result.aggregated_score
                if hasattr(result, "aggregated_score")
                else result.score
            )

            reranked_result = RerankingResult(
                doc_id=result.doc_id,
                content=result.content,
                original_score=score,
                reranked_score=score,
                original_rank=idx,
                final_rank=idx,
                metadata=result.metadata,
            )
            reranked.append(reranked_result)

        return reranked

    def _rerank_by_diversity(
        self,
        results: list[Any],
        metrics: RerankingMetrics,
        diversity_weight: float = 0.3,
        **kwargs,
    ) -> list[RerankingResult]:
        """다양성을 고려한 재순위지정"""
        reranked: list[RerankingResult] = []
        selected_results: list[Any] = []

        def get_score(r):
            return r.aggregated_score if hasattr(r, "aggregated_score") else r.score

        sorted_results = sorted(results, key=lambda x: get_score(x), reverse=True)

        for idx, result in enumerate(sorted_results):
            # 다양성 페널티 계산
            penalty = self._diversity_calc.calculate_diversity_penalty(
                result, selected_results, diversity_weight
            )

            # 재순위지정 점수
            score = get_score(result)
            reranked_score = score * (1.0 - penalty)

            reranked_result = RerankingResult(
                doc_id=result.doc_id,
                content=result.content,
                original_score=score,
                reranked_score=reranked_score,
                original_rank=idx,
                final_rank=0,  # 나중에 설정
                reranking_factors={"diversity_penalty": penalty},
                metadata=result.metadata,
            )
            reranked.append(reranked_result)
            selected_results.append(result)

        # 재순위지정 점수 기반 정렬
        reranked.sort(key=lambda x: x.reranked_score, reverse=True)

        # 최종 순위 설정 및 변화 추적
        for idx, r in enumerate(reranked):
            r.final_rank = idx
            rank_change = abs(r.original_rank - idx)
            metrics.rank_changes += rank_change
            metrics.max_rank_change = max(metrics.max_rank_change, rank_change)

        return reranked

    def _rerank_by_recency(
        self, results: list[Any], metrics: RerankingMetrics
    ) -> list[RerankingResult]:
        """시간성을 고려한 재순위지정"""
        reranked = []

        for idx, result in enumerate(results):
            # 최근 결과에 더 높은 점수
            timestamp = (
                result.aggregation_timestamp
                if hasattr(result, "aggregation_timestamp")
                else result.timestamp
            )
            age_factor = 1.0 - (time.time() - timestamp) / (24 * 3600)
            age_factor = max(0.5, age_factor)  # 최소 0.5

            score = (
                result.aggregated_score
                if hasattr(result, "aggregated_score")
                else result.score
            )
            reranked_score = score * age_factor

            reranked_result = RerankingResult(
                doc_id=result.doc_id,
                content=result.content,
                original_score=score,
                reranked_score=reranked_score,
                original_rank=idx,
                final_rank=0,
                reranking_factors={"age_factor": age_factor},
                metadata=result.metadata,
            )
            reranked.append(reranked_result)

        # 정렬 및 순위 업데이트
        reranked.sort(key=lambda x: x.reranked_score, reverse=True)

        for idx, r in enumerate(reranked):
            r.final_rank = idx
            metrics.rank_changes += abs(r.original_rank - idx)

        return reranked

    def _rerank_by_cross_encoder(
        self,
        results: list[Any],
        query_text: str | None,
        metrics: RerankingMetrics,
        **kwargs,
    ) -> list[RerankingResult]:
        """교차 인코더 기반 재순위지정"""
        reranked = []

        for idx, result in enumerate(results):
            # 쿼리 관련성 점수 계산
            relevance_score = self._relevance_scorer.calculate_relevance_score(
                result, query_text or "", ScoringMethod.SIGMOID
            )

            # 원본 점수와 관련성 점수 병합
            score = (
                result.aggregated_score
                if hasattr(result, "aggregated_score")
                else result.score
            )
            reranked_score = 0.6 * score + 0.4 * relevance_score

            reranked_result = RerankingResult(
                doc_id=result.doc_id,
                content=result.content,
                original_score=score,
                reranked_score=reranked_score,
                original_rank=idx,
                final_rank=0,
                reranking_factors={"relevance": relevance_score},
                metadata=result.metadata,
            )
            reranked.append(reranked_result)

        # 정렬 및 순위 업데이트
        reranked.sort(key=lambda x: x.reranked_score, reverse=True)

        for idx, r in enumerate(reranked):
            r.final_rank = idx
            metrics.rank_changes += abs(r.original_rank - idx)

        return reranked

    def _rerank_by_mmr(
        self,
        results: list[Any],
        metrics: RerankingMetrics,
        mmr_lambda: float = 0.5,
        **kwargs,
    ) -> list[RerankingResult]:
        """Maximal Marginal Relevance 기반 재순위지정"""
        reranked: list[RerankingResult] = []
        selected_indices: list[int] = []
        remaining_indices = list(range(len(results)))

        def get_score(r):
            return r.aggregated_score if hasattr(r, "aggregated_score") else r.score

        while remaining_indices and len(reranked) < len(results):
            best_idx = None
            best_mmr = -float("inf")

            for i in remaining_indices:
                # 관련성 점수
                relevance = get_score(results[i])

                # 다양성 페널티
                diversity_penalty = 0.0
                for selected_i in selected_indices:
                    # 간단한 유사도 (점수 차이)
                    similarity = 1.0 - abs(
                        get_score(results[i]) - get_score(results[selected_i])
                    )
                    diversity_penalty = max(diversity_penalty, similarity)

                # MMR = λ * relevance - (1-λ) * diversity_penalty
                mmr_score = (
                    mmr_lambda * relevance - (1 - mmr_lambda) * diversity_penalty
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx is not None:
                result = results[best_idx]
                reranked_result = RerankingResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    original_score=get_score(result),
                    reranked_score=best_mmr,
                    original_rank=best_idx,
                    final_rank=len(reranked),
                    reranking_factors={"mmr": best_mmr},
                    metadata=result.metadata,
                )
                reranked.append(reranked_result)
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                metrics.rank_changes += abs(best_idx - len(reranked) + 1)

        return reranked

    def _rerank_by_fusion(
        self,
        results: list[Any],
        query_text: str | None,
        metrics: RerankingMetrics,
        **kwargs,
    ) -> list[RerankingResult]:
        """여러 전략 통합한 재순위지정"""
        # 1. 점수 기반
        score_reranked, _ = self.rerank(results, strategy=RerankerStrategy.SCORE_ONLY)

        # 2. 관련성 기반
        relevance_reranked, _ = self.rerank(
            results, query_text, strategy=RerankerStrategy.CROSS_ENCODER
        )

        # 3. 다양성 고려
        diversity_reranked, _ = self.rerank(
            results, strategy=RerankerStrategy.DIVERSITY
        )

        # 통합 (평균 순위)
        rank_map: dict[str, list[int]] = {}
        for r in score_reranked:
            rank_map[r.doc_id] = [r.final_rank]
        for r in relevance_reranked:
            if r.doc_id in rank_map:
                rank_map[r.doc_id].append(r.final_rank)
        for r in diversity_reranked:
            if r.doc_id in rank_map:
                rank_map[r.doc_id].append(r.final_rank)

        # 평균 순위로 재정렬
        fusion_results = []

        def get_score(r):
            return r.aggregated_score if hasattr(r, "aggregated_score") else r.score

        for idx, result in enumerate(results):
            ranks = rank_map.get(result.doc_id, [idx])
            avg_rank = sum(ranks) / len(ranks)
            avg_score = 1.0 - (avg_rank / len(results))  # 순위를 점수로 변환

            reranked_result = RerankingResult(
                doc_id=result.doc_id,
                content=result.content,
                original_score=get_score(result),
                reranked_score=avg_score,
                original_rank=idx,
                final_rank=0,
                reranking_factors={"fusion_rank": avg_rank},
                metadata=result.metadata,
            )
            fusion_results.append(reranked_result)

        # 정렬
        fusion_results.sort(key=lambda x: x.reranked_score, reverse=True)

        for idx, r in enumerate(fusion_results):
            r.final_rank = idx
            metrics.rank_changes += abs(r.original_rank - idx)

        return fusion_results
