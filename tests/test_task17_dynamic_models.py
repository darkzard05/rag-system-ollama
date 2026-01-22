"""
Task 17: 동적 모델 로딩 테스트
- 모델 레지스트리 (생성, 조회, 메트릭, 비교)
- 동적 모델 로더 (캐싱, 로드/언로드, TTL, 정책)
- 자동 모델 선택기 (선택 전략, 제약, 적응형)

Target: 25+ 테스트 (100% 통과)
"""

import asyncio
import pytest

from src.core.model_registry import (
    ModelRegistry, ModelInfo, ModelTask,
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
# 픽스처
# ============================================================================

@pytest.fixture
def registry():
    """모델 레지스트리 픽스처."""
    reset_model_registry()
    reg = get_model_registry()
    
    models = [
        ModelInfo(
            name="Llama 2", version="7b",
            task_support=[ModelTask.QA, ModelTask.GENERATION],
            memory_mb=4096, latency_ms=500
        ),
        ModelInfo(
            name="Mistral", version="7b",
            task_support=[ModelTask.QA, ModelTask.GENERATION],
            memory_mb=3072, latency_ms=300
        ),
        ModelInfo(
            name="Neural Chat", version="7b",
            task_support=[ModelTask.QA, ModelTask.SUMMARIZATION],
            memory_mb=2048, latency_ms=200
        ),
        ModelInfo(
            name="Orca Mini", version="3b",
            task_support=[ModelTask.CLASSIFICATION],
            memory_mb=1024, latency_ms=100
        ),
    ]
    
    for model in models:
        reg.register_model(model)
    
    return reg


@pytest.fixture
def model_loader():
    """동적 모델 로더."""
    reset_dynamic_model_loader()
    return get_dynamic_model_loader()


@pytest.fixture
def model_selector(registry):
    """모델 선택기."""
    reset_model_selectors()
    return get_model_selector()


@pytest.fixture
def adaptive_selector(registry):
    """적응형 모델 선택기."""
    reset_model_selectors()
    return get_adaptive_model_selector()


# ============================================================================
# Test Group 1: 모델 레지스트리
# ============================================================================

class TestRegistry:
    """모델 레지스트리 테스트 (6개)."""
    
    def test_01_register_model(self, registry):
        """모델 등록."""
        model = registry.get_model("Llama 2", "7b")
        assert model is not None
        assert model.name == "Llama 2"
    
    def test_02_list_by_task(self, registry):
        """작업별 모델 목록."""
        qa_models = registry.list_models_by_task(ModelTask.QA)
        assert len(qa_models) >= 2
    
    def test_03_record_metric(self, registry):
        """메트릭 기록."""
        registry.record_metric("Llama 2:7b", 250, 4000)
        registry.record_metric("Llama 2:7b", 350, 4100)
        
        metrics = registry.get_metrics("Llama 2:7b")
        assert metrics.total_requests == 2
        assert metrics.avg_latency_ms > 0
    
    def test_04_compare_models(self, registry):
        """모델 비교."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        registry.record_metric("Neural Chat:7b", 200, 2000)
        
        comparison = registry.compare_models(
            ["Llama 2:7b", "Mistral:7b", "Neural Chat:7b"],
            metric="latency"
        )
        
        assert len(comparison) == 3
    
    def test_05_unregister_model(self, registry):
        """모델 제거."""
        registry.unregister_model("Orca Mini", "3b")
        model = registry.get_model("Orca Mini", "3b")
        assert model is None
    
    def test_06_list_all_models(self, registry):
        """모든 모델 조회."""
        models = registry.list_all_models()
        assert len(models) >= 3


# ============================================================================
# Test Group 2: 모델 캐시
# ============================================================================

class TestCache:
    """모델 캐시 테스트 (4개)."""
    
    def test_07_cache_put_get(self):
        """캐시 저장/조회."""
        config = ModelCacheConfig(max_models=3)
        cache = ModelCache(config)
        
        model = {"id": "test"}
        cache.put("model1", model)
        assert cache.get("model1") == model
    
    def test_08_cache_eviction(self):
        """LRU 제거."""
        config = ModelCacheConfig(max_models=2)
        cache = ModelCache(config)
        
        cache.put("m1", {})
        cache.put("m2", {})
        cache.put("m3", {})  # m1 제거
        
        assert cache.get("m1") is None
        assert cache.get("m2") is not None
    
    def test_09_cache_ttl(self):
        """TTL 만료."""
        config = ModelCacheConfig(ttl_seconds=0.1)
        cache = ModelCache(config)
        
        cache.put("m1", {})
        assert cache.get("m1") is not None
        
        import time
        time.sleep(0.2)
        assert cache.get("m1") is None
    
    def test_10_cache_stats(self):
        """캐시 통계."""
        config = ModelCacheConfig(max_models=3)
        cache = ModelCache(config)
        
        cache.put("m1", {})
        cache.put("m2", {})
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 3


# ============================================================================
# Test Group 3: 동적 모델 로더
# ============================================================================

class TestLoader:
    """동적 모델 로더 테스트 (5개)."""
    
    @pytest.mark.asyncio
    async def test_11_load_model(self, model_loader):
        """모델 로드."""
        model = await model_loader.load_model("llama2")
        assert model is not None
    
    @pytest.mark.asyncio
    async def test_12_cache_model(self, model_loader):
        """모델 캐싱."""
        await model_loader.load_model("mistral")
        assert "mistral" in model_loader.cache._cache
    
    @pytest.mark.asyncio
    async def test_13_unload_model(self, model_loader):
        """모델 언로드."""
        await model_loader.load_model("neural-chat")
        await model_loader.unload_model("neural-chat")
        assert "neural-chat" not in model_loader._loaded_models
    
    @pytest.mark.asyncio
    async def test_14_preload_models(self, model_loader):
        """모델 사전 로드."""
        await model_loader.preload_models(["m1", "m2"])
        assert len(model_loader.get_loaded_models()) == 2
    
    @pytest.mark.asyncio
    async def test_15_auto_unload(self, model_loader):
        """미사용 모델 언로드."""
        await model_loader.preload_models(["m1", "m2", "m3"])
        
        # m1만 자주 사용
        for _ in range(3):
            await model_loader.load_model("m1")
        
        await model_loader.auto_unload_unused(keep_models=1)
        
        loaded = model_loader.get_loaded_models()
        assert len(loaded) == 1


# ============================================================================
# Test Group 4: 모델 비교 및 선택
# ============================================================================

class TestComparator:
    """모델 성능 비교자 테스트 (2개)."""
    
    def test_16_compare_by_metric(self, registry):
        """메트릭 기준 비교."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        comp = ModelPerformanceComparator(registry)
        result = comp.compare_by_metric(
            ["Llama 2:7b", "Mistral:7b"],
            "latency_ms"
        )
        
        assert len(result) == 2
    
    def test_17_rank_models(self, registry):
        """가중치 기반 순위."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        comp = ModelPerformanceComparator(registry)
        result = comp.rank_models(
            ["Llama 2:7b", "Mistral:7b"],
            {"latency_ms": 0.5, "memory_mb": 0.5}
        )
        
        assert len(result) > 0


# ============================================================================
# Test Group 5: 모델 선택기
# ============================================================================

class TestSelector:
    """모델 선택기 테스트 (6개)."""
    
    def test_18_select_fastest(self, registry, model_selector):
        """가장 빠른 모델."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        selected = model_selector.select_model(
            ModelTask.QA,
            SelectionStrategy.FASTEST
        )
        assert selected is not None
    
    def test_19_select_balanced(self, registry, model_selector):
        """균형 모델."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        selected = model_selector.select_model(
            ModelTask.QA,
            SelectionStrategy.BALANCED
        )
        assert selected is not None
    
    def test_20_with_constraint(self, registry, model_selector):
        """제약 조건."""
        constraint = SelectionConstraint(max_memory_mb=2048)
        selected = model_selector.select_model(
            ModelTask.QA,
            constraint=constraint
        )
        assert selected is not None
    
    def test_21_should_switch(self, registry, model_selector):
        """모델 전환 필요 여부."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        model_selector.select_model(ModelTask.QA)
        should_switch = model_selector.should_switch_model(ModelTask.QA)
        
        assert isinstance(should_switch, bool)
    
    def test_22_recommendation(self, registry, model_selector):
        """모델 추천."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        
        rec = model_selector.get_model_recommendation(ModelTask.QA)
        assert "recommended_model" in rec
        assert "should_switch" in rec
    
    def test_23_no_available(self, model_selector):
        """사용 가능한 모델 없음."""
        selected = model_selector.select_model(ModelTask.EMBEDDING)
        assert selected is None


# ============================================================================
# Test Group 6: 적응형 선택기
# ============================================================================

class TestAdaptive:
    """적응형 선택기 테스트 (4개)."""
    
    def test_24_evaluate(self, registry, adaptive_selector):
        """성능 평가 및 적응."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        
        model = adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        assert model is not None
    
    def test_25_threshold(self, registry, adaptive_selector):
        """성능 임계값."""
        adaptive_selector.set_performance_threshold(0.5)
        
        model = adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        assert model is not None
    
    def test_26_switch_history(self, registry, adaptive_selector):
        """전환 이력."""
        adaptive_selector.evaluate_and_adapt(ModelTask.QA)
        
        history = adaptive_selector.get_switch_history(ModelTask.QA)
        assert isinstance(history, list)
    
    def test_27_adaptation_stats(self, adaptive_selector):
        """적응 통계."""
        stats = adaptive_selector.get_adaptation_stats()
        
        assert "total_switches" in stats
        assert "tasks_adapted" in stats


# ============================================================================
# Test Group 7: 통합 및 엣지 케이스
# ============================================================================

class TestIntegration:
    """통합 테스트 (4개)."""
    
    @pytest.mark.asyncio
    async def test_28_end_to_end(self, registry, model_loader, model_selector):
        """엔드-투-엔드 워크플로우."""
        selected = model_selector.select_model(ModelTask.QA)
        assert selected is not None
        
        loaded = await model_loader.load_model(selected)
        assert loaded is not None
    
    @pytest.mark.asyncio
    async def test_29_concurrent_loads(self, model_loader):
        """동시 로드."""
        tasks = [
            model_loader.load_model("m1"),
            model_loader.load_model("m2"),
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)
    
    def test_30_constraint_impossible(self, model_selector):
        """불가능한 제약."""
        constraint = SelectionConstraint(max_memory_mb=256)
        selected = model_selector.select_model(
            ModelTask.QA,
            constraint=constraint
        )
        assert selected is None
    
    def test_31_multiple_selections(self, registry, model_selector):
        """다중 선택."""
        registry.record_metric("Llama 2:7b", 500, 4000)
        registry.record_metric("Mistral:7b", 300, 3000)
        
        for _ in range(3):
            selected = model_selector.select_model(ModelTask.QA)
            assert selected is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
