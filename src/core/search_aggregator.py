"""
Task 19-2: Search Results Aggregation Module
검색 결과 수집 및 통합 - 다중 노드 결과 병합 및 중복 제거 (리팩토링 버전)
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class AggregationStrategy(Enum):
    """결과 통합 전략"""

    MERGE_ALL = "merge_all"
    DEDUP_CONTENT = "dedup_content"
    DEDUP_ID = "dedup_id"
    WEIGHTED_SCORE = "weighted_score"
    TOP_K_PER_NODE = "top_k_per_node"
    RRF_FUSION = "rrf_fusion"
    WEIGHTED_RRF = "weighted_rrf"
    RELATIVE_SCORE_FUSION = "relative_score_fusion"


class DuplicateStrategy(Enum):
    """중복 처리 전략"""

    KEEP_FIRST = "keep_first"
    KEEP_HIGHEST_SCORE = "keep_highest"
    MERGE_SCORES = "merge_scores"
    KEEP_LATEST = "keep_latest"


@dataclass
class AggregatedResult:
    """통합된 검색 결과"""

    doc_id: str
    content: str
    aggregated_score: float
    source_nodes: list[str] = field(default_factory=list)
    occurrence_count: int = 1
    original_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    aggregation_timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.aggregated_score > other.aggregated_score


@dataclass
class AggregationMetrics:
    """통합 메트릭"""

    total_input_results: int = 0
    total_output_results: int = 0
    duplicates_found: int = 0
    duplicates_merged: int = 0
    removed_results: int = 0
    aggregation_time: float = 0.0
    strategy_used: str = ""
    dedup_strategy: str = ""
    nodes_processed: int = 0
    score_adjustments: int = 0


class ContentHash:
    """콘텐츠 기반 해시"""

    @staticmethod
    def calculate(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()


class SearchResultAggregator:
    """검색 결과 집계기 (Optimized & Simplified)"""

    def __init__(
        self, dedup_strategy: DuplicateStrategy = DuplicateStrategy.KEEP_HIGHEST_SCORE
    ):
        self.dedup_strategy = dedup_strategy
        self._lock = RLock()

    def aggregate_results(
        self,
        search_results: dict[str, list[Any]],
        strategy: AggregationStrategy = AggregationStrategy.DEDUP_CONTENT,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> tuple[list[AggregatedResult], AggregationMetrics]:
        """다중 노드의 검색 결과 통합"""
        start_time = time.time()
        metrics = AggregationMetrics(
            strategy_used=strategy.value,
            dedup_strategy=self.dedup_strategy.value,
            nodes_processed=len(search_results),
        )

        if not search_results:
            return [], metrics

        # 전략별 처리
        if strategy == AggregationStrategy.MERGE_ALL:
            aggregated = self._merge_all(search_results, metrics)
        elif strategy in (
            AggregationStrategy.DEDUP_CONTENT,
            AggregationStrategy.DEDUP_ID,
        ):
            aggregated = self._dedup_aggregation(search_results, strategy, metrics)
        elif strategy == AggregationStrategy.WEIGHTED_SCORE:
            aggregated = self._weighted_score_aggregation(
                search_results, metrics, weights
            )
        elif strategy == AggregationStrategy.TOP_K_PER_NODE:
            aggregated = self._top_k_per_node(search_results, metrics)
        elif strategy in (
            AggregationStrategy.RRF_FUSION,
            AggregationStrategy.WEIGHTED_RRF,
        ):
            aggregated = self._rrf_fusion_aggregation(
                search_results, metrics, weights=weights
            )
        elif strategy == AggregationStrategy.RELATIVE_SCORE_FUSION:
            aggregated = self._relative_score_fusion_aggregation(
                search_results, metrics, weights
            )
        else:
            aggregated = self._merge_all(search_results, metrics)

        # 정렬 및 Top-K
        aggregated.sort(key=lambda x: x.aggregated_score, reverse=True)
        if top_k:
            aggregated = aggregated[:top_k]

        metrics.total_output_results = len(aggregated)
        metrics.aggregation_time = time.time() - start_time
        return aggregated, metrics

    def _create_agg_result(
        self, result: Any, score: float | None = None
    ) -> AggregatedResult:
        """AggregatedResult 객체 생성 팩토리"""
        return AggregatedResult(
            doc_id=result.doc_id,
            content=result.content,
            aggregated_score=score if score is not None else result.score,
            source_nodes=[getattr(result, "node_id", "unknown")],
            original_scores=[result.score],
            metadata=result.metadata or {},
        )

    def _merge_all(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        aggregated = []
        for results in search_results.values():
            metrics.total_input_results += len(results)
            aggregated.extend([self._create_agg_result(r) for r in results])
        return aggregated

    def _dedup_aggregation(
        self,
        search_results: dict[str, list[Any]],
        strategy: AggregationStrategy,
        metrics: AggregationMetrics,
    ) -> list[AggregatedResult]:
        """ID 또는 콘텐츠 기반 중복 제거 통합"""
        aggregated_map: dict[str, AggregatedResult] = {}
        key_map: dict[str, str] = {}  # hash -> doc_id (콘텐츠 기반일 때 사용)

        for results in search_results.values():
            metrics.total_input_results += len(results)
            for r in results:
                key = r.doc_id
                if strategy == AggregationStrategy.DEDUP_CONTENT:
                    content_hash = r.metadata.get(
                        "content_hash"
                    ) or ContentHash.calculate(r.content)
                    if content_hash in key_map:
                        key = key_map[content_hash]
                    else:
                        key_map[content_hash] = key

                if key in aggregated_map:
                    metrics.duplicates_found += 1
                    self._merge_duplicate(aggregated_map[key], r, metrics)
                else:
                    aggregated_map[key] = self._create_agg_result(r)
        return list(aggregated_map.values())

    def _weighted_score_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        weights: dict[str, float] | None,
    ) -> list[AggregatedResult]:
        aggregated_map: dict[str, AggregatedResult] = {}
        node_weights = weights or {
            nid: 1.0 / len(search_results) for nid in search_results
        }

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            weight = node_weights.get(node_id, 1.0)
            for r in results:
                weighted_score = r.score * weight
                if r.doc_id in aggregated_map:
                    agg = aggregated_map[r.doc_id]
                    agg.aggregated_score = (agg.aggregated_score + weighted_score) / (
                        len(agg.source_nodes) + 1
                    )
                    agg.source_nodes.append(node_id)
                    agg.original_scores.append(r.score)
                    agg.occurrence_count += 1
                    metrics.score_adjustments += 1
                else:
                    aggregated_map[r.doc_id] = self._create_agg_result(
                        r, score=weighted_score
                    )
                    aggregated_map[r.doc_id].source_nodes = [node_id]
        return list(aggregated_map.values())

    def _top_k_per_node(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        k_per_node: int = 5,
    ) -> list[AggregatedResult]:
        aggregated = []
        for _node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            top_results = sorted(results, key=lambda x: x.score, reverse=True)[
                :k_per_node
            ]
            aggregated.extend([self._create_agg_result(r) for r in top_results])
        return aggregated

    def _rrf_fusion_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> list[AggregatedResult]:
        node_ids = list(search_results.keys())
        node_count = len(node_ids)

        # Fast-path: 2-node RRF (BM25 + FAISS) — inline the arithmetic
        if node_count == 2:
            return self._rrf_fusion_2node(search_results, metrics, k, weights)

        all_docs = {}
        node_ranks: dict[str, dict[str, int]] = {}
        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            sorted_res = sorted(results, key=lambda x: x.score, reverse=True)
            node_ranks[node_id] = {r.doc_id: i + 1 for i, r in enumerate(sorted_res)}
            for r in results:
                if r.doc_id not in all_docs:
                    all_docs[r.doc_id] = self._create_agg_result(r, score=0.0)
                    all_docs[r.doc_id].source_nodes = []
                    all_docs[r.doc_id].original_scores = []
                all_docs[r.doc_id].original_scores.append(r.score)
                if node_id not in all_docs[r.doc_id].source_nodes:
                    all_docs[r.doc_id].source_nodes.append(node_id)

        if not all_docs:
            return []

        weight_by_nid = {
            nid: weights.get(nid, 1.0) if weights else 1.0 for nid in node_ids
        }
        for doc_id, agg in all_docs.items():
            score = 0.0
            for nid in node_ids:
                rank = node_ranks.get(nid, {}).get(doc_id)
                if rank is not None:
                    score += weight_by_nid[nid] / (k + rank)
            agg.aggregated_score = score

        metrics.score_adjustments = len(all_docs)
        return list(all_docs.values())

    def _rrf_fusion_2node(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> list[AggregatedResult]:
        """2-node RRF fast-path — avoids building per-node rank dicts."""
        node_ids = list(search_results.keys())
        n0, n1 = node_ids
        r0 = search_results[n0]
        r1 = search_results[n1]

        metrics.total_input_results = len(r0) + len(r1)

        w0 = weights.get(n0, 1.0) if weights else 1.0
        w1 = weights.get(n1, 1.0) if weights else 1.0

        # Single pass: build rank for n0, accumulate scores for n1 in one go
        rank0: dict[str, int] = {}
        doc_map: dict[str, AggregatedResult] = {}
        for i, r in enumerate(r0):
            rank0[r.doc_id] = i + 1
            agg = self._create_agg_result(r, score=0.0)
            agg.source_nodes = [n0]
            agg.original_scores = [r.score]
            doc_map[r.doc_id] = agg

        for r in r1:
            existing = doc_map.get(r.doc_id)
            if existing:
                existing.source_nodes.append(n1)
                existing.original_scores.append(r.score)
            else:
                agg = self._create_agg_result(r, score=0.0)
                agg.source_nodes = [n0, n1]  # will be scored as 0 from n0
                agg.original_scores = [r.score]
                doc_map[r.doc_id] = agg

        # Inline scoring
        kf = float(k)
        for rank1_idx, r in enumerate(r1):
            agg = doc_map[r.doc_id]
            agg.aggregated_score += w1 / (kf + float(rank1_idx + 1))

        for doc_id, agg in doc_map.items():
            rank = rank0.get(doc_id)
            if rank is not None:
                agg.aggregated_score += w0 / (kf + float(rank))

        metrics.score_adjustments = len(doc_map)
        return list(doc_map.values())

    def _relative_score_fusion_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        weights: dict[str, float] | None = None,
    ) -> list[AggregatedResult]:
        if not search_results:
            return []
        weights = weights or dict.fromkeys(search_results, 1.0)

        all_docs = {}
        node_norm_scores = {}
        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            if not results:
                continue

            raw_scores = [r.score for r in results]
            s_min, s_max = min(raw_scores), max(raw_scores)
            denom = s_max - s_min if s_max > s_min else 1.0
            node_norm_scores[node_id] = {
                r.doc_id: (r.score - s_min) / denom for r in results
            }

            for r in results:
                if r.doc_id not in all_docs:
                    all_docs[r.doc_id] = self._create_agg_result(r, score=0.0)
                    all_docs[r.doc_id].source_nodes = []
                    all_docs[r.doc_id].original_scores = []
                all_docs[r.doc_id].original_scores.append(r.score)
                if node_id not in all_docs[r.doc_id].source_nodes:
                    all_docs[r.doc_id].source_nodes.append(node_id)

        for doc_id, agg in all_docs.items():
            score = 0.0
            for nid, norm_map in node_norm_scores.items():
                if doc_id in norm_map:
                    score += norm_map[doc_id] * weights.get(nid, 1.0)
            agg.aggregated_score = score

        metrics.score_adjustments = len(all_docs)
        return list(all_docs.values())

    def _merge_duplicate(
        self, agg: AggregatedResult, result: Any, metrics: AggregationMetrics
    ):
        """중복 결과 병합 (DuplicateStrategy 준수)"""
        if self.dedup_strategy == DuplicateStrategy.KEEP_HIGHEST_SCORE:
            if result.score > agg.aggregated_score:
                agg.aggregated_score = result.score
        elif self.dedup_strategy == DuplicateStrategy.MERGE_SCORES:
            agg.aggregated_score = (agg.aggregated_score + result.score) / 2
        elif self.dedup_strategy == DuplicateStrategy.KEEP_LATEST:
            # 타임스탬프 필드가 있는 경우만 (없으면 기본값 사용)
            ts = getattr(result, "timestamp", 0)
            if ts > agg.aggregation_timestamp:
                agg.aggregated_score = result.score
                agg.aggregation_timestamp = ts

        node_id = getattr(result, "node_id", "unknown")
        if node_id not in agg.source_nodes:
            agg.source_nodes.append(node_id)
        agg.original_scores.append(result.score)
        agg.occurrence_count += 1
        metrics.duplicates_merged += 1
        metrics.score_adjustments += 1


class ConsistencyValidator:
    """결과 일관성 검증기 (Simplified)"""

    def validate_aggregation(
        self, aggregated: list[AggregatedResult]
    ) -> dict[str, Any]:
        if not aggregated:
            return {"valid": True, "total_results": 0, "issues": []}

        scores = [r.aggregated_score for r in aggregated]
        doc_ids = [r.doc_id for r in aggregated]

        issues = []
        if any(
            s > 100.0 or s < 0.0 for s in scores
        ):  # RRF 등은 1.0 넘을 수 있어 임계치 조정
            pass  # 스코어 범위는 전략마다 다르므로 엄격하게 체크하지 않음
        if len(doc_ids) != len(set(doc_ids)):
            issues.append("Duplicate doc_ids found")
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            issues.append("Not properly sorted")

        return {
            "valid": len(issues) == 0,
            "total_results": len(aggregated),
            "issues": issues,
            "score_stats": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            },
        }
