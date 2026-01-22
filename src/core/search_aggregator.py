"""
Task 19-2: Search Results Aggregation Module
검색 결과 수집 및 통합 - 다중 노드 결과 병합 및 중복 제거
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime
import time
from threading import RLock
import hashlib


class AggregationStrategy(Enum):
    """결과 통합 전략"""
    MERGE_ALL = "merge_all"          # 모든 결과 병합
    DEDUP_CONTENT = "dedup_content"  # 콘텐츠 기반 중복 제거
    DEDUP_ID = "dedup_id"            # ID 기반 중복 제거
    WEIGHTED_SCORE = "weighted_score"  # 가중 스코어 재계산
    TOP_K_PER_NODE = "top_k_per_node"  # 노드당 top-k


class DuplicateStrategy(Enum):
    """중복 처리 전략"""
    KEEP_FIRST = "keep_first"        # 첫 번째 유지
    KEEP_HIGHEST_SCORE = "keep_highest"  # 가장 높은 스코어 유지
    MERGE_SCORES = "merge_scores"    # 스코어 병합 (평균)
    KEEP_LATEST = "keep_latest"      # 최신 타임스탬프 유지


@dataclass
class AggregatedResult:
    """통합된 검색 결과"""
    doc_id: str
    content: str
    aggregated_score: float
    source_nodes: List[str] = field(default_factory=list)
    occurrence_count: int = 1
    original_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
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
        """두 해시의 유사도 확인"""
        if hash1 == hash2:
            return True
        # 간단한 문자 매칭 기반 유사도
        matches = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
        return matches / len(hash1) >= threshold


class SearchResultAggregator:
    """검색 결과 집계기"""
    
    def __init__(self, dedup_strategy: DuplicateStrategy = DuplicateStrategy.KEEP_HIGHEST_SCORE):
        """
        Args:
            dedup_strategy: 중복 제거 전략
        """
        self.dedup_strategy = dedup_strategy
        self._lock = RLock()
        self._content_hashes: Dict[str, str] = {}
        self._id_map: Dict[str, str] = {}
    
    def aggregate_results(
        self,
        search_results: Dict[str, List[Any]],
        strategy: AggregationStrategy = AggregationStrategy.DEDUP_CONTENT,
        top_k: Optional[int] = None
    ) -> Tuple[List[AggregatedResult], AggregationMetrics]:
        """
        다중 노드의 검색 결과 통합
        
        Args:
            search_results: {node_id: [results]} 형태의 검색 결과
            strategy: 통합 전략
            top_k: 상위 k개 결과 (None이면 모두)
            
        Returns:
            (통합된 결과, 메트릭)
        """
        start_time = time.time()
        metrics = AggregationMetrics(
            strategy_used=strategy.value,
            dedup_strategy=self.dedup_strategy.value,
            nodes_processed=len(search_results)
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
        self,
        search_results: Dict[str, List[Any]],
        metrics: AggregationMetrics
    ) -> List[AggregatedResult]:
        """모든 결과 단순 병합"""
        aggregated = []
        
        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            
            for result in results:
                agg_result = AggregatedResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    aggregated_score=result.score,
                    source_nodes=[result.node_id],
                    original_scores=[result.score],
                    metadata=result.metadata
                )
                aggregated.append(agg_result)
        
        return aggregated
    
    def _dedup_by_content(
        self,
        search_results: Dict[str, List[Any]],
        metrics: AggregationMetrics
    ) -> List[AggregatedResult]:
        """콘텐츠 기반 중복 제거"""
        aggregated_map: Dict[str, AggregatedResult] = {}
        
        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            
            for result in results:
                content_hash = ContentHash.calculate(result.content)
                
                # 기존 콘텐츠 해시 확인
                existing_id = None
                for hash_key, agg_id in self._content_hashes.items():
                    if ContentHash.similar_hash(content_hash, hash_key):
                        existing_id = agg_id
                        break
                
                if existing_id:
                    # 중복 병합
                    metrics.duplicates_found += 1
                    agg = aggregated_map[existing_id]
                    
                    self._merge_duplicate(agg, result, metrics)
                
                else:
                    # 새로운 결과
                    with self._lock:
                        self._content_hashes[content_hash] = result.doc_id
                    
                    agg_result = AggregatedResult(
                        doc_id=result.doc_id,
                        content=result.content,
                        aggregated_score=result.score,
                        source_nodes=[result.node_id],
                        original_scores=[result.score],
                        metadata=result.metadata
                    )
                    aggregated_map[result.doc_id] = agg_result
        
        return list(aggregated_map.values())
    
    def _dedup_by_id(
        self,
        search_results: Dict[str, List[Any]],
        metrics: AggregationMetrics
    ) -> List[AggregatedResult]:
        """ID 기반 중복 제거"""
        aggregated_map: Dict[str, AggregatedResult] = {}
        
        for node_id, results in search_results.items():
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
                        metadata=result.metadata
                    )
                    aggregated_map[result.doc_id] = agg_result
        
        return list(aggregated_map.values())
    
    def _weighted_score_aggregation(
        self,
        search_results: Dict[str, List[Any]],
        metrics: AggregationMetrics
    ) -> List[AggregatedResult]:
        """가중 스코어 재계산"""
        aggregated_map: Dict[str, AggregatedResult] = {}
        node_weights = {node_id: 1.0 for node_id in search_results}
        
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
                        metadata=result.metadata
                    )
                    aggregated_map[result.doc_id] = agg_result
        
        return list(aggregated_map.values())
    
    def _top_k_per_node(
        self,
        search_results: Dict[str, List[Any]],
        metrics: AggregationMetrics
    ) -> List[AggregatedResult]:
        """노드당 top-k 결과 취합"""
        aggregated = []
        k_per_node = 5  # 노드당 5개
        
        for node_id, results in search_results.items():
            metrics.total_input_results += len(results)
            
            # 노드별로 top-k 선택
            top_results = sorted(results, key=lambda x: x.score, reverse=True)[:k_per_node]
            
            for result in top_results:
                agg_result = AggregatedResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    aggregated_score=result.score,
                    source_nodes=[node_id],
                    original_scores=[result.score],
                    metadata=result.metadata
                )
                aggregated.append(agg_result)
        
        return aggregated
    
    def _merge_duplicate(self, agg: AggregatedResult, result: Any, metrics: AggregationMetrics):
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
        
        elif self.dedup_strategy == DuplicateStrategy.KEEP_LATEST:
            # 최신 타임스탬프 유지
            if result.timestamp > agg.aggregation_timestamp:
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
    
    def validate_aggregation(self, aggregated: List[AggregatedResult]) -> Dict[str, Any]:
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
                'valid': len(issues) == 0,
                'total_results': len(aggregated),
                'issues': issues,
                'score_stats': {
                    'min': min([r.aggregated_score for r in aggregated]) if aggregated else 0,
                    'max': max([r.aggregated_score for r in aggregated]) if aggregated else 0,
                    'avg': sum([r.aggregated_score for r in aggregated]) / len(aggregated) if aggregated else 0
                }
            }


class ResultDeduplicator:
    """검색 결과 중복 제거기"""
    
    def __init__(self):
        self._lock = RLock()
    
    def find_duplicates(
        self,
        results: List[Any],
        similarity_threshold: float = 0.8
    ) -> List[Tuple[int, int, float]]:
        """
        중복 결과 찾기
        
        Args:
            results: 검색 결과
            similarity_threshold: 유사도 임계값
            
        Returns:
            [(index1, index2, similarity), ...] 형태의 중복 쌍
        """
        duplicates = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = self._calculate_similarity(results[i], results[j])
                if similarity >= similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def _calculate_similarity(self, result1: Any, result2: Any) -> float:
        """두 결과의 유사도 계산"""
        # ID 기반 유사도
        if result1.doc_id == result2.doc_id:
            return 1.0
        
        # 콘텐츠 기반 유사도 (간단한 구현)
        hash1 = ContentHash.calculate(result1.content)
        hash2 = ContentHash.calculate(result2.content)
        
        matches = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
        return matches / len(hash1)
