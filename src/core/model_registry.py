"""
모델 레지스트리 - Ollama 모델 등록 및 메타데이터 관리.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from threading import RLock
import time

logger = logging.getLogger(__name__)


class ModelTask(Enum):
    """모델이 지원하는 작업."""
    QA = "qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    """모델 정보."""
    name: str
    version: str = "latest"
    task_support: List[ModelTask] = field(default_factory=list)
    memory_mb: int = 0
    latency_ms: int = 0
    quantization_level: Optional[str] = None  # 4-bit, 8-bit, 16-bit
    context_length: int = 2048
    parameters: int = 0  # 모델 파라미터 수
    quality_score: float = 0.8  # 0-1
    source: str = "ollama"
    last_used: float = 0.0


@dataclass
class ModelMetrics:
    """모델 성능 메트릭."""
    model_name: str
    total_requests: int = 0
    total_latency_ms: float = 0.0
    total_memory_mb: float = 0.0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    avg_memory_mb: float = 0.0
    success_rate: float = 1.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    latency_history: List[float] = field(default_factory=list)


class ModelRegistry:
    """모델 레지스트리 - 모든 사용 가능 모델 관리."""
    
    def __init__(self):
        self._lock = RLock()
        self._models: Dict[str, ModelInfo] = {}
        self._metrics: Dict[str, ModelMetrics] = {}
        self._task_index: Dict[ModelTask, Set[str]] = {
            task: set() for task in ModelTask
        }
    
    def register_model(self, model_info: ModelInfo):
        """모델 등록."""
        with self._lock:
            model_key = f"{model_info.name}:{model_info.version}"
            
            if model_key in self._models:
                logger.warning(f"모델 이미 등록됨: {model_key}")
                return
            
            self._models[model_key] = model_info
            
            # 작업별 인덱스 업데이트
            for task in model_info.task_support:
                self._task_index[task].add(model_key)
            
            # 메트릭 초기화
            self._metrics[model_key] = ModelMetrics(model_name=model_key)
            
            logger.info(f"모델 등록 완료: {model_key}")
    
    def unregister_model(self, name: str, version: str = "latest"):
        """모델 등록 해제."""
        with self._lock:
            model_key = f"{name}:{version}"
            
            if model_key not in self._models:
                logger.warning(f"등록되지 않은 모델: {model_key}")
                return
            
            model_info = self._models[model_key]
            
            # 작업별 인덱스 제거
            for task in model_info.task_support:
                self._task_index[task].discard(model_key)
            
            del self._models[model_key]
            del self._metrics[model_key]
            
            logger.info(f"모델 등록 해제: {model_key}")
    
    def get_model(self, name: str, version: str = "latest") -> Optional[ModelInfo]:
        """모델 조회."""
        with self._lock:
            model_key = f"{name}:{version}"
            return self._models.get(model_key)
    
    def get_model_by_key(self, model_key: str) -> Optional[ModelInfo]:
        """모델 키로 조회."""
        with self._lock:
            return self._models.get(model_key)
    
    def list_all_models(self) -> List[ModelInfo]:
        """모든 모델 조회."""
        with self._lock:
            return list(self._models.values())
    
    def list_models_by_task(self, task: ModelTask) -> List[ModelInfo]:
        """작업별 모델 조회."""
        with self._lock:
            model_keys = self._task_index.get(task, set())
            return [self._models[key] for key in model_keys]
    
    def list_models_sorted_by(self, sort_key: str) -> List[ModelInfo]:
        """정렬하여 모델 조회.
        
        Args:
            sort_key: 'latency', 'memory', 'quality'
        """
        with self._lock:
            models = list(self._models.values())
        
        if sort_key == "latency":
            return sorted(models, key=lambda m: m.latency_ms)
        elif sort_key == "memory":
            return sorted(models, key=lambda m: m.memory_mb)
        elif sort_key == "quality":
            return sorted(models, key=lambda m: m.quality_score, reverse=True)
        else:
            return models
    
    def record_metric(
        self,
        model_key: str,
        latency_ms: float,
        memory_mb: float,
        error: bool = False,
    ):
        """성능 메트릭 기록."""
        with self._lock:
            if model_key not in self._metrics:
                return
            
            metrics = self._metrics[model_key]
            metrics.total_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.total_memory_mb += memory_mb
            
            if error:
                metrics.error_count += 1
            
            # 히스토리 유지 (최근 1000개)
            metrics.latency_history.append(latency_ms)
            if len(metrics.latency_history) > 1000:
                metrics.latency_history.pop(0)
            
            # 평균/최대/최소 업데이트
            metrics.avg_latency_ms = metrics.total_latency_ms / metrics.total_requests
            metrics.max_latency_ms = max(metrics.latency_history) if metrics.latency_history else 0
            metrics.min_latency_ms = min(metrics.latency_history) if metrics.latency_history else 0
            metrics.avg_memory_mb = metrics.total_memory_mb / metrics.total_requests
            metrics.success_rate = (metrics.total_requests - metrics.error_count) / metrics.total_requests
            
            # P50, P99 계산
            if metrics.latency_history:
                sorted_latencies = sorted(metrics.latency_history)
                metrics.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
                metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            # 마지막 사용 시간 업데이트
            if model_key in self._models:
                self._models[model_key].last_used = time.time()
    
    def get_metrics(self, model_key: str) -> Optional[ModelMetrics]:
        """메트릭 조회."""
        with self._lock:
            return self._metrics.get(model_key)
    
    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """모든 메트릭 조회."""
        with self._lock:
            return dict(self._metrics)
    
    def compare_models(
        self,
        model_keys: List[str],
        metric: str = "latency"
    ) -> List[tuple]:
        """모델 비교.
        
        Returns:
            [(model_key, metric_value), ...] 정렬된 리스트
        """
        results = []
        
        with self._lock:
            for model_key in model_keys:
                metrics = self._metrics.get(model_key)
                if not metrics:
                    continue
                
                if metric == "latency":
                    value = metrics.avg_latency_ms
                elif metric == "memory":
                    value = metrics.avg_memory_mb
                elif metric == "quality":
                    model = self._models.get(model_key)
                    value = model.quality_score if model else 0
                elif metric == "success_rate":
                    value = metrics.success_rate
                else:
                    continue
                
                results.append((model_key, value))
        
        # 숫자가 작을수록 좋은 메트릭 (latency, memory)
        if metric in ["latency", "memory"]:
            return sorted(results, key=lambda x: x[1])
        else:
            # 숫자가 클수록 좋은 메트릭
            return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_least_used_model(self) -> Optional[str]:
        """가장 적게 사용된 모델 조회."""
        with self._lock:
            if not self._models:
                return None
            
            return min(
                self._models.keys(),
                key=lambda k: self._models[k].last_used
            )
    
    def get_model_stats_summary(self) -> Dict[str, Any]:
        """모델 통계 요약."""
        with self._lock:
            total_models = len(self._models)
            total_requests = sum(m.total_requests for m in self._metrics.values())
            avg_success_rate = sum(m.success_rate for m in self._metrics.values()) / len(self._metrics) if self._metrics else 0
            
            return {
                "total_models": total_models,
                "total_requests": total_requests,
                "avg_success_rate": avg_success_rate,
                "tasks_supported": len([t for t in self._task_index if self._task_index[t]]),
            }
    
    def reset_metrics(self, model_key: Optional[str] = None):
        """메트릭 리셋."""
        with self._lock:
            if model_key:
                if model_key in self._metrics:
                    self._metrics[model_key] = ModelMetrics(model_name=model_key)
            else:
                for key in self._metrics:
                    self._metrics[key] = ModelMetrics(model_name=key)


# 전역 인스턴스
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """전역 모델 레지스트리 조회."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


def reset_model_registry():
    """모델 레지스트리 리셋 (테스트용)."""
    global _registry_instance
    _registry_instance = None
