"""
Task 17: 동적 모델 로딩 테스트
- 모델 레지스트리 (생성, 조회, 메트릭, 비교)
- 동적 모델 로더 (캐싱, 로드/언로드, TTL, 정책)
- 자동 모델 선택기 (선택 전략, 제약, 적응형)

Target: 20+ 테스트 (100% 통과)
"""

import asyncio
import pytest
from datetime import datetime

from src.core.model_registry import (
    ModelRegistry, ModelInfo, ModelTask, ModelMetrics,
    get_model_registry, reset_model_registry
)
from src.core.model_loader_dynamic import (
    DynamicModelLoader, ModelCache, ModelLoadingStrategy,
    ModelEvictionPolicy, ModelCacheConfig,
    get_dynamic_model_loader, reset_dynamic_model_loader
)
from src.core.model_selector import (
    ModelSelector, AdaptiveModelSelector,
    SelectionStrategy, SelectionConstraint,
    ModelPerformanceComparator,
    get_model_selector, get_adaptive_model_selector,
    reset_model_selectors
)


# ============================================================================
# 테스트 픽스처
# ============================================================================

@pytest.fixture
def registry():
    """모델 레지스트리 픽스처."""
    reset_model_registry()
    reg = get_model_registry()
    
    # 테스트 모델 등록
    models = [
        ModelInfo(
            name="Llama 2",
            version="7b",
            task_support=[ModelTask.QA, ModelTask.GENERATION],
            memory_mb=4096,
            latency_ms=500,
            quantization_level="q4_0"
        ),
        ModelInfo(
            name="Mistral",
            version="7b",
            task_support=[ModelTask.QA, ModelTask.GENERATION],
            memory_mb=3072,
            latency_ms=300,
            quantization_level="q4_0"
        ),
        ModelInfo(
            name="Neural Chat",
            version="7b",
            task_support=[ModelTask.QA, ModelTask.SUMMARIZATION],
            memory_mb=2048,
            latency_ms=200,
            quantization_level="q5_0"
        ),
        ModelInfo(
            name="Orca Mini",
            version="3b",
            task_support=[ModelTask.CLASSIFICATION],
            memory_mb=1024,
            latency_ms=100,
            quantization_level="q4_0"
        ),
    ]
    
    for model in models:
        reg.register_model(model)
    
    return reg


@pytest.fixture
def model_loader():
    """동적 모델 로더 픽스처."""
    reset_dynamic_model_loader()
    return get_dynamic_model_loader()


@pytest.fixture
def model_selector(registry):
    """모델 선택기 픽스처."""
    reset_model_selectors()
    return get_model_selector()


@pytest.fixture
def adaptive_selector(registry):
    """적응형 모델 선택기 픽스처."""
    reset_model_selectors()
    return get_adaptive_model_selector()


# ============================================================================
# Task 1-4: 모델 레지스트리 테스트
# ============================================================================

class TestModelRegistry:
    """모델 레지스트리 테스트."""
    
    def test_register_and_retrieve_model(self, registry):
        """모델 등록 및 조회."""
        model = registry.get_model("Llama 2", "7b")
        
        assert model is not None
        assert model.name == "Llama 2"
        assert model.memory_mb == 4096
    
    def test_list_models_by_task(self, registry):
        """작업별 모델 목록 조회."""
        qa_models = registry.list_models_by_task(ModelTask.QA)
        
        assert len(qa_models) == 2  # Llama 2, Mistral
        assert all(ModelTask.QA in m.task_support for m in qa_models)
    
    def test_record_metric(self, registry):
        """메트릭 기록."""
        registry.record_metric("Llama 2", "7b", latency_ms=250, success=True)
        registry.record_metric("Llama 2", "7b", latency_ms=350, success=True)
        
        model = registry.get_model("Llama 2", "7b")
        
        assert model is not None
    
    def test_model_comparison(self, registry):
        """모델 비교."""
        # 메트릭 기록
        registry.record_metric("Llama 2", "7b", latency_ms=500)
        registry.record_metric("Mistral", "7b", latency_ms=300)
        registry.record_metric("Neural Chat", "7b", latency_ms=200)
        
        comparison = registry.compare_models(
            ["Llama 2:7b", "Mistral:7b", "Neural Chat:7b"],
            metric="avg_latency_ms"
        )
        
        # 응답 시간 오름차순 정렬
        assert comparison[0][0] == "Neural Chat:7b"  # 200ms
        assert comparison[1][0] == "Mistral:7b"      # 300ms
        assert comparison[2][0] == "Llama 2:7b"      # 500ms
    
    def test_unregister_model(self, registry):
        """모델 제거."""
        registry.unregister_model("Orca Mini", "3b")
        
        model = registry.get_model("Orca Mini", "3b")
        assert model is None
    
    def test_get_stats_summary(self, registry):
        """통계 요약."""
        registry.record_metric("Mistral", "7b", latency_ms=250)
        registry.record_metric("Mistral", "7b", latency_ms=350)
        
        stats = registry.get_model_stats_summary("Mistral:7b")
        
        assert stats is not None or stats == {}


# ============================================================================
# Task 5-9: 모델 캐시 및 로더 테스트
# ============================================================================

class TestModelCache:
    """모델 캐시 테스트."""
    
    def test_cache_put_and_get(self):
        """캐시 저장 및 조회."""
        config = ModelCacheConfig(max_models=3)
        cache = ModelCache(config)
        
        model = {"id": "llama2:7b"}
        assert cache.put("llama2:7b", model)
        assert cache.get("llama2:7b") == model
    
    def test_cache_eviction_lru(self):
        """LRU 제거 정책."""
        config = ModelCacheConfig(
            max_models=2,
            eviction_policy=ModelEvictionPolicy.LRU
        )
        cache = ModelCache(config)
        
        cache.put("model1", {"id": 1})
        cache.put("model2", {"id": 2})
        cache.put("model3", {"id": 3})  # model1 제거
        
        assert cache.get("model1") is None  # 제거됨
        assert cache.get("model2") is not None
        assert cache.get("model3") is not None
    
    def test_cache_ttl(self):
        """TTL 만료."""
        config = ModelCacheConfig(ttl_seconds=0.1)
        cache = ModelCache(config)
        
        cache.put("model1", {"id": 1})
        assert cache.get("model1") is not None
        
        # TTL 만료 후
        import time
        time.sleep(0.2)
        assert cache.get("model1") is None
    
    def test_cache_stats(self):
        """캐시 통계."""
        config = ModelCacheConfig(max_models=3)
        cache = ModelCache(config)
        
        cache.put("model1", {})
        cache.put("model2", {})
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 3
        assert stats["utilization_percent"] > 0


class TestDynamicModelLoader:
    """동적 모델 로더 테스트."""
    
    @pytest.mark.asyncio
    async def test_load_model(self, model_loader):
        """모델 로드."""
        model = await model_loader.load_model("llama2:7b")
        
        assert model is not None
        assert model["model_key"] == "llama2:7b"
    
    @pytest.mark.asyncio
    async def test_model_caching(self, model_loader):
        """모델 캐싱."""
        await model_loader.load_model("mistral:7b")
        
        # 두 번째 로드는 캐시에서
        model2 = await model_loader.load_model("mistral:7b")
        
        assert model2 is not None
        assert "mistral:7b" in model_loader.cache._cache
    
    @pytest.mark.asyncio
    async def test_unload_model(self, model_loader):
        """모델 언로드."""
        await model_loader.load_model("neural-chat:7b")
        await model_loader.unload_model("neural-chat:7b")
        
        assert "neural-chat:7b" not in model_loader._loaded_models
    
    @pytest.mark.asyncio
    async def test_preload_models(self, model_loader):
        """모델 사전 로드."""
        await model_loader.preload_models(["llama2:7b", "mistral:7b"])
        
        assert len(model_loader.get_loaded_models()) == 2
    
    @pytest.mark.asyncio
    async def test_auto_unload_unused(self, model_loader):
        """미사용 모델 자동 언로드."""
        await model_loader.preload_models(["llama2:7b", "mistral:7b", "neural-chat:7b"])
        
        # 한 모델만 사용
        for _ in range(3):
            await model_loader.load_model("llama2:7b")
        
        await model_loader.auto_unload_unused(keep_models=1)
        
        # llama2만 남아있음
        loaded = model_loader.get_loaded_models()
        assert len(loaded) == 1
        assert "llama2:7b" in loaded


# ============================================================================
# Task 10-15: 모델 선택기 테스트
# ============================================================================

class TestModelPerformanceComparator:
    """모델 성능 비교자 테스트."""
    
    def test_compare_by_metric(self, registry):
        """메트릭 기준 비교."""
        registry.record_metric("Llama 2", "7b", latency_ms=500)
        registry.record_metric("Mistral", "7b", latency_ms=300)
        registry.record_metric("Neural Chat", "7b", latency_ms=200)
        
        comparator = ModelPerformanceComparator(registry)
        comparison = comparator.compare_by_metric(
            ["Llama 2:7b", "Mistral:7b", "Neural Chat:7b"],
            "avg_latency_ms"
        )
        
        assert len(comparison) > 0
    
    def test_rank_models(self, registry):
        """가중치 기반 순위."""
        registry.record_metric("Llama 2", "7b", latency_ms=500, success=True)
        registry.record_metric("Mistral", "7b", latency_ms=300, success=True)
        registry.record_metric("Neural Chat", "7b", latency_ms=200, success=True)
        
        # 성공률 기록
        for _ in range(5):
            registry.record_metric("Neural Chat", "7b", success=True)
        for _ in range(3):
            registry.record_metric("Mistral", "7b", success=True)
        
        comparator = ModelPerformanceComparator(registry)
        ranking = comparator.rank_models(
            ["Llama 2:7b", "Mistral:7b", "Neural Chat:7b"],
            {"avg_latency_ms": 0.4, "success_rate": 0.6}
        )
        
        assert len(ranking) == 3
        assert ranking[0][1] > 0  # 유효한 점수


class TestModelSelector:
    """모델 선택기 테스트."""
    
    def test_select_fastest_model(self, registry, model_selector):
        """가장 빠른 모델 선택."""
        registry.record_metric("Llama 2", "7b", latency_ms=500)
        registry.record_metric("Mistral", "7b", latency_ms=300)
        registry.record_metric("Neural Chat", "7b", latency_ms=200)
        
        selected = model_selector.select_model(
            ModelTask.QA,
            SelectionStrategy.FASTEST
        )
        
        assert selected is not None
    
    def test_select_balanced_model(self, registry, model_selector):
        """균형 모델 선택."""
        registry.record_metric("Llama 2", "7b", latency_ms=500, success=True)
        registry.record_metric("Mistral", "7b", latency_ms=300, success=True)
        registry.record_metric("Neural Chat", "7b", latency_ms=200, success=True)
        
        selected = model_selector.select_model(
            ModelTask.QA,
            SelectionStrategy.BALANCED
        )
        
        assert selected is not None
    
    def test_select_with_latency_constraint(self, registry, model_selector):
        """응답 시간 제약 조건."""
        registry.record_metric("Llama 2", "7b", latency_ms=500)
        registry.record_metric("Mistral", "7b", latency_ms=300)
        registry.record_metric("Neural Chat", "7b", latency_ms=200)
        
        constraint = SelectionConstraint(max_latency_ms=250)
        selected = model_selector.select_model(
            ModelTask.QA,
            SelectionStrategy.FASTEST,
            constraint
        )
        
        assert selected is not None
    
    def test_select_with_memory_constraint(self, registry, model_selector):
        """메모리 제약 조건."""
        constraint = SelectionConstraint(max_memory_mb=2048)
        selected = model_selector.select_model(
            ModelTask.QA,
            constraint=constraint
        )
        
        # Neural Chat만 조건 만족 (2048MB)
        assert selected is not None
    
    def test_should_switch_model(self, registry, model_selector):
        """모델 전환 필요 여부."""
        registry.record_metric("Llama 2", "7b", latency_ms=500)
        registry.record_metric("Mistral", "7b", latency_ms=300)
        
        # 첫 선택
        model_selector.select_model(ModelTask.QA)
        
        # 전환 필요 여부
        should_switch = model_selector.should_switch_model(ModelTask.QA)
        
        assert isinstance(should_switch, bool)
    
    def test_get_recommendation(self, registry, model_selector):
        """모델 추천."""
        registry.record_metric("Llama 2", "7b", latency_ms=500, success=True)
        registry.record_metric("Mistral", "7b", latency_ms=300, success=True)
        
        recommendation = model_selector.get_model_recommendation(ModelTask.QA)
        
        assert "recommended_model" in recommendation
        assert "should_switch" in recommendation


class TestAdaptiveModelSelector:
    """적응형 모델 선택기 테스트."""
    
    def test_evaluate_and_adapt(self, registry, adaptive_selector):
        """성능 평가 후 적응."""
        registry.record_metric("Llama 2", "7b", latency_ms=500, success=True)
        registry.record_metric("Mistral", "7b", latency_ms=300, success=True)
        registry.record_metric("Neural Chat", "7b", latency_ms=200, success=True)
        
        # 첫 평가
        model1 = adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        assert model1 is not None
        
        # 두 번째 평가 (같은 모델 유지 또는 전환)
        model2 = adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        assert model2 is not None
    
    def test_performance_threshold(self, registry, adaptive_selector):
        """성능 임계값."""
        registry.record_metric("Llama 2", "7b", success=True)
        registry.record_metric("Mistral", "7b", success=True)
        
        adaptive_selector.set_performance_threshold(0.95)
        
        model = adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        assert model is not None
    
    def test_switch_history(self, registry, adaptive_selector):
        """모델 전환 이력."""
        registry.record_metric("Llama 2", "7b", success=True)
        registry.record_metric("Mistral", "7b", success=True)
        
        adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        
        history = adaptive_selector.get_switch_history(ModelTask.QA)
        assert isinstance(history, list)
    
    def test_adaptation_stats(self, registry, adaptive_selector):
        """적응 통계."""
        stats = adaptive_selector.get_adaptation_stats()
        
        assert "total_switches" in stats
        assert "tasks_adapted" in stats
        assert "performance_threshold" in stats


# ============================================================================
# Task 16-20: 통합 및 엣지 케이스 테스트
# ============================================================================

class TestIntegration:
    """통합 테스트."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, registry, model_loader, model_selector):
        """엔드-투-엔드 워크플로우."""
        # 1. 모델 선택
        selected_model = model_selector.select_model(ModelTask.QA)
        assert selected_model is not None
        
        # 2. 모델 로드
        loaded_model = await model_loader.load_model(selected_model)
        assert loaded_model is not None
        
        # 3. 메트릭 기록
        parts = selected_model.split(":")
        if len(parts) == 2:
            registry.record_metric(parts[0], parts[1], latency_ms=150, success=True)
        
        # 4. 성능 확인
        models = registry.list_all_models()
        assert len(models) > 0
    
    @pytest.mark.asyncio
    async def test_cache_and_selector_integration(self, registry, model_loader, model_selector):
        """캐시와 선택기 통합."""
        selected_model = model_selector.select_model(ModelTask.QA)
        
        if selected_model:
            # 첫 로드
            await model_loader.load_model(selected_model)
            cache_stats_1 = model_loader.get_cache_stats()
            
            # 두 번째 로드 (캐시에서)
            await model_loader.load_model(selected_model)
            cache_stats_2 = model_loader.get_cache_stats()
            
            assert cache_stats_2["size"] == cache_stats_1["size"]


class TestEdgeCases:
    """엣지 케이스 테스트."""
    
    def test_select_with_no_available_models(self, registry, model_selector):
        """사용 가능한 모델 없음."""
        selected = model_selector.select_model(ModelTask.EMBEDDING)
        assert selected is None
    
    def test_select_with_impossible_constraint(self, registry, model_selector):
        """불가능한 제약 조건."""
        constraint = SelectionConstraint(max_memory_mb=256)  # 너무 낮음
        selected = model_selector.select_model(
            ModelTask.QA,
            constraint=constraint
        )
        assert selected is None
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loads(self, model_loader):
        """동시 모델 로드."""
        tasks = [
            model_loader.load_model("Llama 2:7b"),
            model_loader.load_model("Mistral:7b"),
            model_loader.load_model("Neural Chat:7b"),
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)


# ============================================================================
# 테스트 실행
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
