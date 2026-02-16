"""
Task 19-2: Search Results Aggregation Module
검색 결과 수집 및 통합 - 다중 노드 결과 병합 및 중복 제거
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class AggregationStrategy(Enum):
    """결과 통합 전략"""

    MERGE_ALL = "merge_all"  # 모든 결과 병합
    DEDUP_CONTENT = "dedup_content"  # 콘텐츠 기반 중복 제거
    DEDUP_ID = "dedup_id"  # ID 기반 중복 제거
    WEIGHTED_SCORE = "weighted_score"  # 가중 스코어 재계산
    TOP_K_PER_NODE = "top_k_per_node"  # 노드당 top-k
    RRF_FUSION = "rrf_fusion"  # [최적화] Reciprocal Rank Fusion (NumPy 벡터화)
    WEIGHTED_RRF = "weighted_rrf"  # [최신] 가중치가 적용된 RRF
    RELATIVE_SCORE_FUSION = (
        "relative_score_fusion"  # [최신] 상대 점수 기반 융합 (Min-Max Scaling)
    )


class DuplicateStrategy(Enum):
    """중복 처리 전략"""

    KEEP_FIRST = "keep_first"  # 첫 번째 유지
    KEEP_HIGHEST_SCORE = "keep_highest"  # 가장 높은 스코어 유지
    MERGE_SCORES = "merge_scores"  # 스코어 병합 (평균)
    KEEP_LATEST = "keep_latest"  # 최신 타임스탬프 유지


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
        """점수 기반 비교 (내림차순)"""
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
        """콘텐츠의 SHA256 해시 계산"""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def similar_hash(hash1: str, hash2: str, threshold: float = 0.9) -> bool:
        """
        두 해시의 동일 여부 확인 (SHA256은 유사도 측정이 불가능하므로 완전 일치만 체크)
        """
        return hash1 == hash2


class SearchResultAggregator:
    """검색 결과 집계기"""

    def __init__(
        self, dedup_strategy: DuplicateStrategy = DuplicateStrategy.KEEP_HIGHEST_SCORE
    ):
        """
        Args:
            dedup_strategy: 중복 제거 전략
        """
        self.dedup_strategy = dedup_strategy
        self._lock = RLock()
        self._content_hashes: dict[str, str] = {}
        self._id_map: dict[str, str] = {}

    def aggregate_results(
        self,
        search_results: dict[str, list[Any]],
        strategy: AggregationStrategy = AggregationStrategy.DEDUP_CONTENT,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> tuple[list[AggregatedResult], AggregationMetrics]:
        """
        다중 노드의 검색 결과 통합

        Args:
            search_results: {node_id: [results]} 형태의 검색 결과
            strategy: 통합 전략
            top_k: 상위 k개 결과 (None이면 모두)
            weights: 노드별 가중치 (전략에 따라 사용됨)

        Returns:
            (통합된 결과, 메트릭)
        """
        start_time = time.time()
        metrics = AggregationMetrics(
            strategy_used=strategy.value,
            dedup_strategy=self.dedup_strategy.value,
            nodes_processed=len(search_results),
        )

        with self._lock:
            self._content_hashes.clear()
            self._id_map.clear()

        # 결과 통합 전략별 처리
        if strategy == AggregationStrategy.MERGE_ALL:
            aggregated = self._merge_all(search_results, metrics)

        elif strategy == AggregationStrategy.DEDUP_CONTENT:
            aggregated = self._dedup_by_content(search_results, metrics)

        elif strategy == AggregationStrategy.DEDUP_ID:
            aggregated = self._dedup_by_id(search_results, metrics)

        elif strategy == AggregationStrategy.WEIGHTED_SCORE:
            aggregated = self._weighted_score_aggregation(search_results, metrics)

        elif strategy == AggregationStrategy.TOP_K_PER_NODE:
            aggregated = self._top_k_per_node(search_results, metrics)

        elif strategy == AggregationStrategy.RRF_FUSION:
            aggregated = self._rrf_fusion_aggregation(search_results, metrics)

        elif strategy == AggregationStrategy.WEIGHTED_RRF:
            aggregated = self._weighted_rrf_aggregation(
                search_results, metrics, weights=weights
            )

        elif strategy == AggregationStrategy.RELATIVE_SCORE_FUSION:
            aggregated = self._relative_score_fusion_aggregation(
                search_results, metrics, weights=weights
            )

        else:
            aggregated = self._merge_all(search_results, metrics)

        # 정렬
        aggregated.sort(key=lambda x: x.aggregated_score, reverse=True)

        # top-k 적용
        if top_k:
            aggregated = aggregated[:top_k]

        metrics.total_output_results = len(aggregated)
        metrics.aggregation_time = time.time() - start_time

        return aggregated, metrics

    def _merge_all(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        """모든 결과 단순 병합"""
        aggregated = []

        for _node_id, results in search_results.items():
            metrics.total_input_results += len(results)

            for result in results:
                agg_result = AggregatedResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    aggregated_score=result.score,
                    source_nodes=[result.node_id],
                    original_scores=[result.score],
                    metadata=result.metadata,
                )
                aggregated.append(agg_result)

        return aggregated

    def _dedup_by_content(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        """콘텐츠 기반 중복 제거 (O(N) 최적화)"""
        aggregated_map: dict[str, AggregatedResult] = {}
        # [최적화] 해시를 키로 사용하여 즉시 조회
        content_hash_to_id: dict[str, str] = {}

        for _node_id, results in search_results.items():
            metrics.total_input_results += len(results)

            for result in results:
                # [최적화] 메타데이터에 미리 계산된 해시가 있으면 사용, 없으면 실시간 계산 (Fallback)
                content_hash = result.metadata.get("content_hash")
                if not content_hash:
                    content_hash = ContentHash.calculate(result.content)

                # [최적화] 딕셔너리 룩업으로 중복 확인 (O(1))
                existing_id = content_hash_to_id.get(content_hash)

                if existing_id:
                    # 중복 병합
                    metrics.duplicates_found += 1
                    agg = aggregated_map[existing_id]
                    self._merge_duplicate(agg, result, metrics)
                else:
                    # 새로운 결과 등록
                    content_hash_to_id[content_hash] = result.doc_id

                    agg_result = AggregatedResult(
                        doc_id=result.doc_id,
                        content=result.content,
                        aggregated_score=result.score,
                        source_nodes=[result.node_id],
                        original_scores=[result.score],
                        metadata=result.metadata,
                    )
                    aggregated_map[result.doc_id] = agg_result

        return list(aggregated_map.values())

    def _dedup_by_id(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        """ID 기반 중복 제거"""
        aggregated_map: dict[str, AggregatedResult] = {}

        for _node_id, results in search_results.items():
            metrics.total_input_results += len(results)

            for result in results:
                if result.doc_id in aggregated_map:
                    # 중복 병합
                    metrics.duplicates_found += 1
                    agg = aggregated_map[result.doc_id]
                    self._merge_duplicate(agg, result, metrics)

                else:
                    # 새로운 결과
                    agg_result = AggregatedResult(
                        doc_id=result.doc_id,
                        content=result.content,
                        aggregated_score=result.score,
                        source_nodes=[result.node_id],
                        original_scores=[result.score],
                        metadata=result.metadata,
                    )
                    aggregated_map[result.doc_id] = agg_result

        return list(aggregated_map.values())

    def _weighted_score_aggregation(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        """가중 스코어 재계산"""
        aggregated_map: dict[str, AggregatedResult] = {}
        node_weights = dict.fromkeys(search_results, 1.0)

        # 노드 가중치 정규화
        total_nodes = len(search_results)
        for node_id in node_weights:
            node_weights[node_id] = 1.0 / total_nodes

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            weight = node_weights[node_id]

            for result in results:
                if result.doc_id in aggregated_map:
                    # 스코어 재계산
                    metrics.score_adjustments += 1
                    agg = aggregated_map[result.doc_id]

                    weighted_score = result.score * weight
                    agg.aggregated_score = (
                        agg.aggregated_score + weighted_score
                    ) / len(agg.source_nodes + [node_id])

                    agg.source_nodes.append(node_id)
                    agg.original_scores.append(result.score)
                    agg.occurrence_count += 1

                else:
                    # 새로운 결과
                    agg_result = AggregatedResult(
                        doc_id=result.doc_id,
                        content=result.content,
                        aggregated_score=result.score * weight,
                        source_nodes=[node_id],
                        original_scores=[result.score],
                        metadata=result.metadata,
                    )
                    aggregated_map[result.doc_id] = agg_result

        return list(aggregated_map.values())

    def _top_k_per_node(
        self, search_results: dict[str, list[Any]], metrics: AggregationMetrics
    ) -> list[AggregatedResult]:
        """노드당 top-k 결과 취합"""
        aggregated = []
        k_per_node = 5  # 노드당 5개

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)

            # 노드별로 top-k 선택
            top_results = sorted(results, key=lambda x: x.score, reverse=True)[
                :k_per_node
            ]

            for result in top_results:
                agg_result = AggregatedResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    aggregated_score=result.score,
                    source_nodes=[node_id],
                    original_scores=[result.score],
                    metadata=result.metadata,
                )
                aggregated.append(agg_result)

        return aggregated

    def _rrf_fusion_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        k: int = 60,
    ) -> list[AggregatedResult]:
        """
        [최적화] Reciprocal Rank Fusion (RRF) 통합 (NumPy 벡터화)
        전략: 1 / (k + rank) 점수를 통합하여 스케일이 다른 결과들을 공정하게 병합
        """
        import numpy as np

        all_doc_map = {}
        # 각 노드별 문서 순위 맵 {node_id: {doc_id: rank}}
        node_ranks = {}

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            # 점수 내림차순 정렬 후 순위 부여
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            ranks = {res.doc_id: i + 1 for i, res in enumerate(sorted_results)}
            node_ranks[node_id] = ranks

            # 모든 유니크 문서와 정보 수집
            for res in sorted_results:
                if res.doc_id not in all_doc_map:
                    all_doc_map[res.doc_id] = {
                        "content": res.content,
                        "metadata": res.metadata,
                        "original_scores": [],
                        "source_nodes": [],
                    }
                all_doc_map[res.doc_id]["original_scores"].append(res.score)
                all_doc_map[res.doc_id]["source_nodes"].append(node_id)

        # RRF 점수 계산 (NumPy 벡터화)
        doc_ids = list(all_doc_map.keys())
        num_docs = len(doc_ids)
        num_nodes = len(search_results)

        if num_docs == 0:
            return []

        # (num_docs, num_nodes) 행렬 생성하여 순위 저장 (기본값 infinity)
        rank_matrix = np.full((num_docs, num_nodes), np.inf)

        node_id_to_idx = {node_id: i for i, node_id in enumerate(search_results.keys())}

        for i, doc_id in enumerate(doc_ids):
            for node_id, ranks in node_ranks.items():
                if doc_id in ranks:
                    rank_matrix[i, node_id_to_idx[node_id]] = ranks[doc_id]

        # RRF 연산: 1 / (k + rank)
        # np.inf에 대해 1/inf = 0이 되므로 자동 필터링됨
        rrf_scores = np.sum(1.0 / (k + rank_matrix), axis=1)

        aggregated = []
        for i, doc_id in enumerate(doc_ids):
            data = all_doc_map[doc_id]
            agg_result = AggregatedResult(
                doc_id=doc_id,
                content=data["content"],
                aggregated_score=float(rrf_scores[i]),
                source_nodes=data["source_nodes"],
                original_scores=data["original_scores"],
                metadata=data["metadata"],
            )
            aggregated.append(agg_result)

        metrics.score_adjustments = num_docs
        return aggregated

    def _weighted_rrf_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> list[AggregatedResult]:
        """
        [최신] 가중치가 적용된 Reciprocal Rank Fusion (Weighted RRF)
        각 리트리버 노드별 가중치를 적용하여 순위 점수를 계산합니다.
        """
        import numpy as np

        if not search_results:
            return []

        # 기본 가중치는 1.0으로 설정
        if weights is None:
            weights = dict.fromkeys(search_results.keys(), 1.0)

        all_doc_map = {}
        node_ranks = {}
        node_ids = list(search_results.keys())

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            node_ranks[node_id] = {
                res.doc_id: i + 1 for i, res in enumerate(sorted_results)
            }

            for res in sorted_results:
                if res.doc_id not in all_doc_map:
                    all_doc_map[res.doc_id] = {
                        "content": res.content,
                        "metadata": res.metadata,
                        "original_scores": [],
                        "source_nodes": [],
                    }
                all_doc_map[res.doc_id]["original_scores"].append(res.score)
                all_doc_map[res.doc_id]["source_nodes"].append(node_id)

        doc_ids = list(all_doc_map.keys())
        num_docs = len(doc_ids)
        num_nodes = len(node_ids)

        if num_docs == 0:
            return []

        # (num_docs, num_nodes) 랭크 행렬
        rank_matrix = np.full((num_docs, num_nodes), np.inf)
        weight_vector = np.array([weights.get(nid, 1.0) for nid in node_ids])

        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        for i, doc_id in enumerate(doc_ids):
            for node_id, ranks in node_ranks.items():
                if doc_id in ranks:
                    rank_matrix[i, node_id_to_idx[node_id]] = ranks[doc_id]

        # RRF Score = Sum( weight / (k + rank) )
        rrf_scores = np.sum(weight_vector / (k + rank_matrix), axis=1)

        aggregated = []
        for i, doc_id in enumerate(doc_ids):
            data = all_doc_map[doc_id]
            aggregated.append(
                AggregatedResult(
                    doc_id=doc_id,
                    content=data["content"],
                    aggregated_score=float(rrf_scores[i]),
                    source_nodes=data["source_nodes"],
                    original_scores=data["original_scores"],
                    metadata=data["metadata"],
                )
            )

        metrics.score_adjustments = num_docs
        return aggregated

    def _relative_score_fusion_aggregation(
        self,
        search_results: dict[str, list[Any]],
        metrics: AggregationMetrics,
        weights: dict[str, float] | None = None,
    ) -> list[AggregatedResult]:
        """
        [최신] 상대 점수 기반 융합 (Relative Score Fusion / Hybrid Score Fusion)
        각 결과셋의 점수를 Min-Max Scaling으로 정규화한 뒤 가중 합산합니다.
        """
        import numpy as np

        if not search_results:
            return []

        if weights is None:
            weights = dict.fromkeys(search_results.keys(), 1.0)

        all_doc_map = {}
        node_normalized_scores = {}
        node_ids = list(search_results.keys())

        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            if not results:
                continue

            scores = np.array([r.score for r in results])
            min_s, max_s = scores.min(), scores.max()

            # 정규화: (score - min) / (max - min)
            denom = (max_s - min_s) if max_s > min_s else 1.0
            norm_scores = (scores - min_s) / denom

            node_normalized_scores[node_id] = {
                res.doc_id: norm_scores[i] for i, res in enumerate(results)
            }

            for res in results:
                if res.doc_id not in all_doc_map:
                    all_doc_map[res.doc_id] = {
                        "content": res.content,
                        "metadata": res.metadata,
                        "original_scores": [],
                        "source_nodes": [],
                    }
                all_doc_map[res.doc_id]["original_scores"].append(res.score)
                all_doc_map[res.doc_id]["source_nodes"].append(node_id)

        doc_ids = list(all_doc_map.keys())
        aggregated = []

        for doc_id in doc_ids:
            total_score = 0.0
            for node_id in node_ids:
                if (
                    node_id in node_normalized_scores
                    and doc_id in node_normalized_scores[node_id]
                ):
                    total_score += node_normalized_scores[node_id][
                        doc_id
                    ] * weights.get(node_id, 1.0)

            data = all_doc_map[doc_id]
            aggregated.append(
                AggregatedResult(
                    doc_id=doc_id,
                    content=data["content"],
                    aggregated_score=total_score,
                    source_nodes=data["source_nodes"],
                    original_scores=data["original_scores"],
                    metadata=data["metadata"],
                )
            )

        metrics.score_adjustments = len(aggregated)
        return aggregated

    def _merge_duplicate(
        self, agg: AggregatedResult, result: Any, metrics: AggregationMetrics
    ):
        """중복 결과 병합"""
        if self.dedup_strategy == DuplicateStrategy.KEEP_FIRST:
            # 첫 번째만 유지 (기존 agg 유지)
            pass

        elif self.dedup_strategy == DuplicateStrategy.KEEP_HIGHEST_SCORE:
            # 가장 높은 스코어 유지
            metrics.score_adjustments += 1
            if result.score > agg.aggregated_score:
                agg.aggregated_score = result.score

        elif self.dedup_strategy == DuplicateStrategy.MERGE_SCORES:
            # 평균 스코어
            metrics.score_adjustments += 1
            agg.aggregated_score = (agg.aggregated_score + result.score) / 2

        elif (
            self.dedup_strategy == DuplicateStrategy.KEEP_LATEST
            and result.timestamp > agg.aggregation_timestamp
        ):
            # 최신 타임스탬프 유지
            agg.aggregated_score = result.score
            agg.aggregation_timestamp = result.timestamp

        # 공통 처리
        if result.node_id not in agg.source_nodes:
            agg.source_nodes.append(result.node_id)
        agg.original_scores.append(result.score)
        agg.occurrence_count += 1
        metrics.duplicates_merged += 1


class ConsistencyValidator:
    """결과 일관성 검증기"""

    def __init__(self):
        self._lock = RLock()

    def validate_aggregation(
        self, aggregated: list[AggregatedResult]
    ) -> dict[str, Any]:
        """
        통합 결과 검증

        Args:
            aggregated: 통합된 결과

        Returns:
            검증 결과 정보
        """
        with self._lock:
            issues = []

            # 스코어 범위 확인
            if aggregated:
                scores = [r.aggregated_score for r in aggregated]
                min_score = min(scores)
                max_score = max(scores)

                if max_score > 1.0 or min_score < 0.0:
                    issues.append(f"Score out of range: [{min_score}, {max_score}]")

            # 중복 ID 확인
            doc_ids = [r.doc_id for r in aggregated]
            if len(doc_ids) != len(set(doc_ids)):
                issues.append("Duplicate doc_ids found in results")

            # 정렬 확인
            for i in range(len(aggregated) - 1):
                if aggregated[i].aggregated_score < aggregated[i + 1].aggregated_score:
                    issues.append("Results not properly sorted by score")
                    break

            return {
                "valid": len(issues) == 0,
                "total_results": len(aggregated),
                "issues": issues,
                "score_stats": {
                    "min": min([r.aggregated_score for r in aggregated])
                    if aggregated
                    else 0,
                    "max": max([r.aggregated_score for r in aggregated])
                    if aggregated
                    else 0,
                    "avg": sum([r.aggregated_score for r in aggregated])
                    / len(aggregated)
                    if aggregated
                    else 0,
                },
            }


class ResultDeduplicator:
    """검색 결과 중복 제거기"""

    def __init__(self):
        self._lock = RLock()

    def find_duplicates(
        self, results: list[Any], similarity_threshold: float = 0.8
    ) -> list[tuple[int, int, float]]:
        """
        중복 결과 찾기 (최적화 버전)

        Args:
            results: 검색 결과
            similarity_threshold: 유사도 임계값

        Returns:
            [(index1, index2, similarity), ...] 형태의 중복 쌍
        """
        if not results:
            return []

        # [최적화] 해시 미리 계산하여 루프 내 오버헤드 제거
        hashes = [
            r.metadata.get("content_hash") or ContentHash.calculate(r.content)
            for r in results
        ]
        duplicates = []

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # 1. ID 우선 비교
                if results[i].doc_id == results[j].doc_id:
                    duplicates.append((i, j, 1.0))
                    continue

                # 2. 해시 완전 일치 확인 (SHA256 특성상 유사도 의미 없음)
                if hashes[i] == hashes[j]:
                    duplicates.append((i, j, 1.0))

        return duplicates

    def _calculate_similarity(self, result1: Any, result2: Any) -> float:
        """기존 메서드 유지 (호환성용)"""
        if result1.doc_id == result2.doc_id:
            return 1.0
        return 1.0 if result1.content == result2.content else 0.0
