"""
자동 모델 선택자 - 성능, 메모리, 속도 기반 모델 선택 및 전환.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import RLock

from src.core.model_registry import ModelRegistry, ModelTask, get_model_registry

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """모델 선택 전략."""
    FASTEST = "fastest"        # 가장 빠른 모델
    MOST_ACCURATE = "accurate" # 가장 정확한 모델
    BALANCED = "balanced"      # 속도-정확도 균형
    MEMORY_EFFICIENT = "memory"# 메모리 효율적


@dataclass
class SelectionConstraint:
    """모델 선택 제약 조건."""
    max_latency_ms: Optional[float] = None  # 최대 응답 시간
    max_memory_mb: Optional[float] = None   # 최대 메모리
    min_quality_score: Optional[float] = None  # 최소 품질 점수 (0-1)
    required_quantization: Optional[str] = None  # 필수 양자화 타입


class ModelPerformanceComparator:
    """모델 성능 비교자."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
        self._lock = RLock()
    
    def compare_by_metric(
        self,
        model_keys: List[str],
        metric_name: str = "latency_ms",
    ) -> List[Tuple[str, float]]:
        """메트릭 기준 모델 비교.
        
        Args:
            model_keys: 비교할 모델 키 목록
            metric_name: 비교 메트릭 (latency_ms, memory_mb, quality_score 등)
        
        Returns:
            (모델_키, 메트릭값) 정렬된 튜플 리스트 (낮을수록 좋음)
        """
        results = []
        
        for model_key in model_keys:
            model = self.registry.get_model_by_key(model_key)
            if not model:
                continue
            
            if metric_name == "latency_ms":
                value = model.latency_ms
            elif metric_name == "memory_mb":
                value = model.memory_mb
            elif metric_name == "quality_score":
                value = model.quality_score
            else:
                logger.warning(f"메트릭 없음: {metric_name}")
                continue
            
            results.append((model_key, value))
        
        # 오름차순 정렬 (낮을수록 좋음)
        results.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
        
        return results
    
    def rank_models(
        self,
        model_keys: List[str],
        weights: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """가중치 기반 모델 순위.
        
        Args:
            model_keys: 모델 키 목록
            weights: 메트릭별 가중치 (합 = 1.0)
                예: {"latency_ms": 0.4, "quality_score": 0.6}
        
        Returns:
            (모델_키, 점수) 정렬된 튜플 리스트 (높을수록 좋음)
        """
        if not model_keys:
            return []
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight == 0:
            return []
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # 메트릭별 정규화된 값 계산
        metric_scores = {}  # model_key -> {metric -> normalized_score}
        
        for metric_name in weights.keys():
            comparisons = self.compare_by_metric(model_keys, metric_name)
            
            if not comparisons:
                continue
            
            # 값 범위 계산
            values = [v for _, v in comparisons if v is not None]
            if not values:
                continue
            
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val > min_val else 1
            
            # 정규화 점수 계산 (0-1, 높을수록 좋음)
            for model_key in model_keys:
                if model_key not in metric_scores:
                    metric_scores[model_key] = {}
                
                # 모델의 메트릭 값 찾기
                metric_value = None
                for m_key, value in comparisons:
                    if m_key == model_key:
                        metric_value = value
                        break
                
                if metric_value is not None:
                    # 점수 반전 (낮은 latency = 높은 점수)
                    normalized_score = 1.0 - ((metric_value - min_val) / range_val)
                    metric_scores[model_key][metric_name] = normalized_score
                else:
                    metric_scores[model_key][metric_name] = 0.0
        
        # 최종 점수 계산
        final_scores = []
        for model_key in model_keys:
            scores = metric_scores.get(model_key, {})
            final_score = sum(
                scores.get(metric, 0.0) * normalized_weights[metric]
                for metric in weights.keys()
            )
            final_scores.append((model_key, final_score))
        
        # 내림차순 정렬 (높을수록 좋음)
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores


class ModelSelector:
    """자동 모델 선택기."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
        self.comparator = ModelPerformanceComparator(self.registry)
        self._lock = RLock()
        self._current_model: Dict[ModelTask, Optional[str]] = {}
    
    def select_model(
        self,
        task: ModelTask,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        constraint: Optional[SelectionConstraint] = None,
    ) -> Optional[str]:
        """작업에 맞는 최적 모델 선택.
        
        Args:
            task: 작업 유형
            strategy: 선택 전략
            constraint: 선택 제약 조건
        
        Returns:
            선택된 모델 키 (선택 불가 시 None)
        """
        # 작업을 지원하는 모델 목록
        available_models = self.registry.list_models_by_task(task)
        
        if not available_models:
            logger.error(f"사용 가능한 모델 없음: {task.value}")
            return None
        
        model_keys = [f"{m.name}:{m.version}" for m in available_models]
        
        # 제약 조건 적용
        if constraint:
            model_keys = self._apply_constraints(model_keys, constraint)
            
            if not model_keys:
                logger.warning(f"제약을 만족하는 모델 없음: {task.value}")
                return None
        
        # 전략 적용
        selected_model = self._select_by_strategy(model_keys, strategy)
        
        if selected_model:
            with self._lock:
                self._current_model[task] = selected_model
            
            logger.info(f"모델 선택: {task.value} -> {selected_model} ({strategy.value})")
        
        return selected_model
    
    def _apply_constraints(
        self,
        model_keys: List[str],
        constraint: SelectionConstraint,
    ) -> List[str]:
        """제약 조건 적용.
        
        Returns:
            제약을 만족하는 모델 키 목록
        """
        filtered = []
        
        for model_key in model_keys:
            model = self.registry.get_model_by_key(model_key)
            if not model:
                continue
            
            # 응답 시간 확인
            if constraint.max_latency_ms is not None:
                latency = model.latency_ms or float('inf')
                if latency > constraint.max_latency_ms:
                    continue
            
            # 메모리 확인
            if constraint.max_memory_mb is not None:
                memory = model.memory_mb or float('inf')
                if memory > constraint.max_memory_mb:
                    continue
            
            # 품질 점수 확인
            if constraint.min_quality_score is not None:
                quality = model.quality_score or 0.0
                if quality < constraint.min_quality_score:
                    continue
            
            # 양자화 확인
            if constraint.required_quantization is not None:
                if model.quantization_level != constraint.required_quantization:
                    continue
            
            filtered.append(model_key)
        
        return filtered
    
    def _select_by_strategy(
        self,
        model_keys: List[str],
        strategy: SelectionStrategy,
    ) -> Optional[str]:
        """전략 기반 모델 선택."""
        if not model_keys:
            return None
        
        if strategy == SelectionStrategy.FASTEST:
            # 가장 빠른 모델
            comparisons = self.comparator.compare_by_metric(
                model_keys, "latency_ms"
            )
            return comparisons[0][0] if comparisons else None
        
        elif strategy == SelectionStrategy.MOST_ACCURATE:
            # 가장 정확한 모델 (품질 점수 높음)
            comparisons = self.comparator.compare_by_metric(
                model_keys, "quality_score"
            )
            # 내림차순 (높을수록 좋음)
            comparisons.reverse()
            return comparisons[0][0] if comparisons else None
        
        elif strategy == SelectionStrategy.BALANCED:
            # 속도-정확도 균형
            weights = {
                "latency_ms": 0.4,
                "quality_score": 0.6,
            }
            ranked = self.comparator.rank_models(model_keys, weights)
            return ranked[0][0] if ranked else None
        
        elif strategy == SelectionStrategy.MEMORY_EFFICIENT:
            # 메모리 효율
            weights = {
                "latency_ms": 0.3,
                "quality_score": 0.3,
                "memory_mb": 0.4,
            }
            ranked = self.comparator.rank_models(model_keys, weights)
            return ranked[0][0] if ranked else None
        
        return model_keys[0]
    
    def get_current_model(self, task: ModelTask) -> Optional[str]:
        """작업의 현재 선택 모델 조회."""
        with self._lock:
            return self._current_model.get(task)
    
    def should_switch_model(
        self,
        task: ModelTask,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> bool:
        """모델 전환 필요 여부 판단.
        
        Returns:
            현재와 다른 더 좋은 모델이 있으면 True
        """
        current_model = self.get_current_model(task)
        new_model = self.select_model(task, strategy)
        
        if not new_model:
            return False
        
        return current_model != new_model
    
    def get_model_recommendation(
        self,
        task: ModelTask,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> Dict:
        """모델 추천 정보.
        
        Returns:
            추천 정보 딕셔너리
        """
        current_model = self.get_current_model(task)
        recommended_model = self.select_model(task, strategy)
        
        result = {
            "task": task.value,
            "current_model": current_model,
            "recommended_model": recommended_model,
            "should_switch": current_model != recommended_model,
        }
        
        if recommended_model:
            model = self.registry.get_model_by_key(recommended_model)
            if model:
                result["metrics"] = {
                    "latency_ms": model.latency_ms,
                    "memory_mb": model.memory_mb,
                    "quality_score": model.quality_score,
                }
        
        return result


class AdaptiveModelSelector:
    """적응형 모델 선택기 - 실시간 성능 기반 전환."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
        self.selector = ModelSelector(self.registry)
        self._lock = RLock()
        self._performance_threshold = 0.8  # 성능 임계값 (0-1)
        self._switch_history: Dict[ModelTask, List[str]] = {}
    
    def evaluate_and_adapt(self, task: ModelTask) -> Optional[str]:
        """성능 평가 후 적응형 모델 선택.
        
        현재 모델의 성능이 임계값 이하면 다른 모델로 전환.
        
        Returns:
            선택된 모델 키
        """
        current_model = self.selector.get_current_model(task)
        
        if not current_model:
            # 첫 선택
            return self.selector.select_model(task)
        
        # 현재 모델 성능 확인
        model = self.registry.get_model_by_key(current_model)
        if not model:
            return self.selector.select_model(task)
        
        # 성능 점수 계산 (0-1, 높을수록 좋음)
        performance_score = model.quality_score or 0.0
        
        # 임계값 이상이면 계속 사용
        if performance_score >= self._performance_threshold:
            logger.debug(f"모델 성능 양호: {current_model} ({performance_score:.2f})")
            return current_model
        
        # 성능 부족 시 다른 모델 찾기
        logger.warning(
            f"모델 성능 부족: {current_model} ({performance_score:.2f}), "
            f"임계값: {self._performance_threshold}"
        )
        
        new_model = self.selector.select_model(task, SelectionStrategy.BALANCED)
        
        if new_model and new_model != current_model:
            with self._lock:
                if task not in self._switch_history:
                    self._switch_history[task] = []
                self._switch_history[task].append(new_model)
            
            logger.info(f"모델 전환: {current_model} -> {new_model}")
        
        return new_model or current_model
    
    def set_performance_threshold(self, threshold: float):
        """성능 임계값 설정."""
        if 0.0 <= threshold <= 1.0:
            self._performance_threshold = threshold
    
    def get_switch_history(self, task: ModelTask) -> List[str]:
        """작업의 모델 전환 이력."""
        with self._lock:
            return self._switch_history.get(task, [])
    
    def get_adaptation_stats(self) -> Dict:
        """적응 통계."""
        with self._lock:
            total_switches = sum(len(v) for v in self._switch_history.values())
            
            return {
                "total_switches": total_switches,
                "tasks_adapted": len(self._switch_history),
                "performance_threshold": self._performance_threshold,
            }


# 전역 인스턴스
_selector_instance: Optional[ModelSelector] = None
_adaptive_selector_instance: Optional[AdaptiveModelSelector] = None


def get_model_selector() -> ModelSelector:
    """전역 모델 선택기 조회."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = ModelSelector()
    return _selector_instance


def get_adaptive_model_selector() -> AdaptiveModelSelector:
    """전역 적응형 모델 선택기 조회."""
    global _adaptive_selector_instance
    if _adaptive_selector_instance is None:
        _adaptive_selector_instance = AdaptiveModelSelector()
    return _adaptive_selector_instance


def reset_model_selectors():
    """모델 선택기 리셋 (테스트용)."""
    global _selector_instance, _adaptive_selector_instance
    _selector_instance = None
    _adaptive_selector_instance = None
