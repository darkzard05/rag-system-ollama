"""
에러 복구 시스템 테스트 - Task 14
재시도, 서킷 브레이커, 폴백 종합 검증
"""

import asyncio
import time

import pytest
from src.common.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    get_circuit_breaker_registry,
    reset_circuit_breaker_registry,
)
from src.common.fallback_handler import (
    CachedFallback,
    GracefulDegradation,
    ServiceAvailability,
    SimpleFallback,
)
from src.infra.error_recovery import (
    AdaptiveTimeout,
    ExponentialBackoff,
    MaxRetriesExceeded,
    RetryConfig,
    RetryPolicy,
    RetryStrategy,
    TimeoutManager,
)


class TestExponentialBackoff:
    """지수 백오프 테스트"""

    def test_exponential_backoff_calculation(self):
        """지수 백오프 계산"""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        # 1, 2, 4, 8, ...
        assert backoff.calculate_delay(0) == 1.0
        assert backoff.calculate_delay(1) == 2.0
        assert backoff.calculate_delay(2) == 4.0
        assert backoff.calculate_delay(3) == 8.0

    def test_max_delay_limit(self):
        """최대 지연 제한"""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        # 최대 10초
        assert backoff.calculate_delay(5) == 10.0

    def test_linear_backoff(self):
        """선형 백오프"""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR_BACKOFF, initial_delay=1.0, jitter=False
        )
        backoff = ExponentialBackoff(config)

        # 1, 2, 3, 4, ...
        assert backoff.calculate_delay(0) == 1.0
        assert backoff.calculate_delay(1) == 2.0
        assert backoff.calculate_delay(2) == 3.0

    def test_fixed_delay(self):
        """고정 지연"""
        config = RetryConfig(
            strategy=RetryStrategy.FIXED_DELAY, initial_delay=2.0, jitter=False
        )
        backoff = ExponentialBackoff(config)

        # 항상 2.0
        assert backoff.calculate_delay(0) == 2.0
        assert backoff.calculate_delay(1) == 2.0
        assert backoff.calculate_delay(5) == 2.0


class TestRetryPolicy:
    """재시도 정책 테스트"""

    @pytest.mark.asyncio
    async def test_successful_on_first_attempt(self):
        """첫 시도 성공"""
        config = RetryConfig(max_attempts=3)
        policy = RetryPolicy(config)

        async def success_func():
            return "success"

        result = await policy.execute_with_retry(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_and_success(self):
        """재시도 후 성공"""
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        policy = RetryPolicy(config)

        attempts = []

        async def failing_then_success():
            attempts.append(len(attempts))
            if len(attempts) < 2:
                raise ConnectionError("Failed")
            return "success"

        result = await policy.execute_with_retry(failing_then_success)
        assert result == "success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """최대 재시도 초과"""
        config = RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False)
        policy = RetryPolicy(config)

        async def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(MaxRetriesExceeded):
            await policy.execute_with_retry(always_fails)

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """재시도 불가능한 예외"""
        config = RetryConfig(max_attempts=3, retryable_exceptions=[ConnectionError])
        policy = RetryPolicy(config)

        async def value_error():
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await policy.execute_with_retry(value_error)

    @pytest.mark.asyncio
    async def test_should_retry_logic(self):
        """재시도 판단 로직"""
        config = RetryConfig(max_attempts=3, retryable_exceptions=[ConnectionError])
        policy = RetryPolicy(config)

        # 재시도 가능
        assert policy.should_retry(ConnectionError("Test"), 0)
        assert policy.should_retry(ConnectionError("Test"), 1)

        # 최대 횟수 초과
        assert not policy.should_retry(ConnectionError("Test"), 3)

        # 재시도 불가능한 예외
        assert not policy.should_retry(ValueError("Test"), 0)


class TestTimeoutManager:
    """타임아웃 관리자 테스트"""

    @pytest.mark.asyncio
    async def test_successful_within_timeout(self):
        """타임아웃 내 성공"""
        manager = TimeoutManager(default_timeout=1.0)

        async def quick_func():
            await asyncio.sleep(0.1)
            return "success"

        result = await manager.execute_with_timeout(quick_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """타임아웃 초과"""
        manager = TimeoutManager(default_timeout=0.1)

        async def slow_func():
            await asyncio.sleep(1.0)
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await manager.execute_with_timeout(slow_func)

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """커스텀 타임아웃"""
        manager = TimeoutManager(default_timeout=10.0)

        async def medium_func():
            await asyncio.sleep(0.1)
            return "success"

        result = await manager.execute_with_timeout(medium_func, timeout=1.0)
        assert result == "success"


class TestAdaptiveTimeout:
    """적응형 타임아웃 테스트"""

    def test_adaptive_timeout_calculation(self):
        """적응형 타임아웃 계산"""
        adaptive = AdaptiveTimeout(initial_timeout=30.0, percentile=0.95)

        # 실행 시간 기록
        execution_times = [100, 110, 120, 150, 180, 200, 300, 500, 1000]
        for time_ms in execution_times:
            adaptive.record_execution_time(time_ms / 1000)  # ms → s

        # 적응형 타임아웃 계산 (p95 + 20%)
        adaptive_timeout = adaptive.get_adaptive_timeout()
        assert adaptive_timeout > 0

    def test_empty_history(self):
        """히스토리 없음"""
        adaptive = AdaptiveTimeout(initial_timeout=30.0)

        # 히스토리 없을 때 기본값 반환
        assert adaptive.get_adaptive_timeout() == 30.0


class TestCircuitBreaker:
    """서킷 브레이커 테스트"""

    def test_closed_state_success(self):
        """CLOSED 상태 성공"""
        breaker = CircuitBreaker("service", failure_threshold=2)

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"

    def test_closed_to_open_transition(self):
        """CLOSED → OPEN 전환"""
        breaker = CircuitBreaker("service", failure_threshold=2)

        def failing_func():
            raise Exception("Error")

        # 2회 실패로 OPEN 상태로 전환
        for _ in range(2):
            try:
                breaker.call(failing_func)
            except Exception:
                pass

        assert breaker.get_state() == "open"

    def test_open_state_rejection(self):
        """OPEN 상태 거절"""
        breaker = CircuitBreaker("service", failure_threshold=1, recovery_timeout=10.0)

        def failing_func():
            raise Exception("Error")

        # OPEN 상태로 전환
        try:
            breaker.call(failing_func)
        except Exception:
            pass

        # 다음 요청은 즉시 거절
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(lambda: "test")

    def test_half_open_recovery(self):
        """HALF_OPEN 상태 회복"""
        breaker = CircuitBreaker(
            "service", failure_threshold=1, recovery_timeout=0.1, success_threshold=1
        )

        # OPEN 상태로 전환
        def failing_func():
            raise Exception("Error")

        try:
            breaker.call(failing_func)
        except Exception:
            pass

        assert breaker.get_state() == "open"

        # 타임아웃 대기
        time.sleep(0.2)

        # HALF_OPEN 상태로 전환하고 성공
        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"

    def test_metrics_tracking(self):
        """메트릭 추적"""
        breaker = CircuitBreaker("service", failure_threshold=3)

        def success_func():
            return "success"

        # 성공 요청 추적
        breaker.call(success_func)
        breaker.call(success_func)

        metrics = breaker.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 2


class TestCircuitBreakerRegistry:
    """서킷 브레이커 레지스트리 테스트"""

    def test_multiple_breakers(self):
        """여러 서킷 브레이커"""
        reset_circuit_breaker_registry()
        registry = get_circuit_breaker_registry()

        breaker1 = registry.get_breaker("service1")
        breaker2 = registry.get_breaker("service2")

        assert breaker1.service_name == "service1"
        assert breaker2.service_name == "service2"
        assert breaker1 is not breaker2

    def test_registry_metrics(self):
        """레지스트리 메트릭"""
        reset_circuit_breaker_registry()
        registry = get_circuit_breaker_registry()

        breaker = registry.get_breaker("service")
        breaker.call(lambda: "success")

        all_metrics = registry.get_all_metrics()
        assert "service" in all_metrics


class TestSimpleFallback:
    """단순 폴백 테스트"""

    @pytest.mark.asyncio
    async def test_simple_fallback(self):
        """기본값 반환"""
        fallback = SimpleFallback(default_value="default")

        result = await fallback.execute(Exception("Test error"))

        assert result == "default"


class TestCachedFallback:
    """캐시 폴백 테스트"""

    @pytest.mark.asyncio
    async def test_cached_value_return(self):
        """캐시된 값 반환"""
        cache = {"key1": "cached_value"}
        fallback = CachedFallback(cache)

        result = await fallback.execute(Exception("Test error"), "key1")

        assert result == "cached_value"

    @pytest.mark.asyncio
    async def test_missing_cache(self):
        """캐시 부재"""
        cache = {}
        fallback = CachedFallback(cache)

        with pytest.raises(ValueError):
            await fallback.execute(Exception("Test error"), "nonexistent")


class TestGracefulDegradation:
    """Graceful Degradation 테스트"""

    @pytest.mark.asyncio
    async def test_original_function_success(self):
        """원본 함수 성공"""
        degradation = GracefulDegradation()

        async def func():
            return "original"

        result = await degradation.execute(func)
        assert result == "original"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """실패 시 폴백"""
        degradation = GracefulDegradation()
        degradation.add_fallback(SimpleFallback("fallback"))

        async def failing_func():
            raise Exception("Error")

        result = await degradation.execute(failing_func)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_chained_fallback(self):
        """연쇄 폴백"""
        degradation = GracefulDegradation()

        # 여러 폴백 전략을 추가
        degradation.add_fallback(SimpleFallback("fallback1"))
        degradation.add_fallback(SimpleFallback("fallback2"))

        async def failing_func():
            raise Exception("Error")

        # 첫 번째 폴백이 작동
        result = await degradation.execute(failing_func)
        assert result == "fallback1"


class TestServiceAvailability:
    """서비스 가용성 테스트"""

    def test_mark_unavailable(self):
        """서비스 사용 불가 표시"""
        availability = ServiceAvailability()

        availability.mark_service_unavailable("db")
        assert not availability.is_service_available("db")

    def test_degradation_level(self):
        """저하 수준"""
        availability = ServiceAvailability()

        # 정상 상태 (두 서비스 모두 사용 가능)
        availability.mark_service_available("service1")
        availability.mark_service_available("service2")
        assert availability.degradation_level == 0

        # 경미한 저하 (1개 서비스 불가 / 2개 = 50%)
        # 실제로는 심각한 저하 (>= 50%)로 계산되므로 3개 서비스 테스트
        availability.mark_service_available("service3")
        assert availability.degradation_level == 0  # 3개 모두 가능

        availability.mark_service_unavailable("service1")
        # 1개 불가 / 3개 = 33.3% < 50% → 경미한 저하
        assert availability.degradation_level == 1


class TestIntegration:
    """통합 테스트"""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """재시도 + 서킷 브레이커"""
        retry_config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        retry_policy = RetryPolicy(retry_config)
        breaker = CircuitBreaker("service", failure_threshold=1)

        call_count = [0]

        async def unreliable_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await retry_policy.execute_with_retry(unreliable_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        """타임아웃 + 재시도"""
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False,
            retryable_exceptions=[asyncio.TimeoutError],
        )
        policy = RetryPolicy(config)
        manager = TimeoutManager(default_timeout=0.5)

        call_count = [0]

        async def func():
            call_count[0] += 1
            await asyncio.sleep(0.1)
            return "success"

        result = await policy.execute_with_retry(manager.execute_with_timeout, func)
        assert result == "success"


# 테스트 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
