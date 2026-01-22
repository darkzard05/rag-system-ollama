"""
폴백 핸들러 - Task 14
Graceful Degradation, 폴백 전략
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FallbackStrategy(ABC):
    """폴백 전략 기본 클래스"""
    
    @abstractmethod
    async def execute(
        self,
        original_error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """
        폴백 실행
        
        Args:
            original_error: 원래 발생한 예외
            *args: 원래 함수의 인자
            **kwargs: 원래 함수의 키워드 인자
        
        Returns:
            폴백 결과
        
        Raises:
            Exception: 폴백 실패 시
        """
        pass


class SimpleFallback(FallbackStrategy):
    """
    단순 폴백
    
    고정된 기본값 반환
    """
    
    def __init__(self, default_value: Any):
        self.default_value = default_value
    
    async def execute(
        self,
        original_error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """기본값 반환"""
        logger.info(
            f"[Fallback] 단순 폴백 실행: {original_error}"
        )
        return self.default_value


class CachedFallback(FallbackStrategy):
    """
    캐시 기반 폴백
    
    캐시된 이전 결과 반환
    """
    
    def __init__(self, cache: Dict[str, Any]):
        self.cache = cache
    
    async def execute(
        self,
        original_error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """캐시된 값 반환"""
        # 캐시 키 생성 (첫 번째 인자 기반)
        cache_key = str(args[0]) if args else "default"
        
        if cache_key in self.cache:
            logger.info(
                f"[Fallback] 캐시 폴백: {cache_key}"
            )
            return self.cache[cache_key]
        
        raise ValueError(f"캐시에 {cache_key}가 없습니다")


class DegradedServiceFallback(FallbackStrategy):
    """
    저하된 서비스 폴백
    
    기능을 제한하여 부분 동작
    """
    
    def __init__(self, degraded_func: Callable):
        self.degraded_func = degraded_func
    
    async def execute(
        self,
        original_error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """저하된 기능 실행"""
        logger.warning(
            f"[Fallback] 저하된 서비스 모드: {original_error}"
        )
        
        if asyncio.iscoroutinefunction(self.degraded_func):
            return await self.degraded_func(*args, **kwargs)
        else:
            return self.degraded_func(*args, **kwargs)


class ChainedFallback(FallbackStrategy):
    """
    연쇄 폴백
    
    여러 폴백 전략을 순차적으로 시도
    """
    
    def __init__(self, strategies: List[FallbackStrategy]):
        self.strategies = strategies
    
    async def execute(
        self,
        original_error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """연쇄 폴백 실행"""
        logger.info(
            f"[Fallback] 연쇄 폴백 시작 ({len(self.strategies)}개 전략)"
        )
        
        for i, strategy in enumerate(self.strategies):
            try:
                logger.debug(
                    f"[Fallback] 전략 {i+1}/{len(self.strategies)} 시도"
                )
                result = await strategy.execute(original_error, *args, **kwargs)
                logger.info(f"[Fallback] 전략 {i+1} 성공")
                return result
            
            except Exception as e:
                logger.warning(
                    f"[Fallback] 전략 {i+1} 실패: {e}"
                )
                continue
        
        # 모든 폴백 실패
        raise Exception("모든 폴백 전략 실패")


@dataclass
class FallbackConfig:
    """폴백 설정"""
    strategy: FallbackStrategy
    error_types: List[type] = field(
        default_factory=lambda: [Exception]
    )
    enabled: bool = True
    log_fallback: bool = True


class GracefulDegradation:
    """
    Graceful 성능 저하
    
    에러 발생 시 기능을 단계적으로 저하
    
    예시:
    1. 원본 함수 실행
    2. 예외 발생 → 캐시된 결과 반환
    3. 캐시 없음 → 간단한 결과 반환
    4. 완전 실패 → 기본값 반환
    """
    
    def __init__(self):
        self.fallback_chain: List[FallbackConfig] = []
    
    def add_fallback(
        self,
        strategy: FallbackStrategy,
        error_types: Optional[List[type]] = None
    ) -> None:
        """폴백 전략 추가"""
        config = FallbackConfig(
            strategy=strategy,
            error_types=error_types or [Exception]
        )
        self.fallback_chain.append(config)
    
    def _should_use_fallback(
        self,
        error: Exception,
        error_types: List[type]
    ) -> bool:
        """폴백 사용 여부"""
        return any(isinstance(error, exc_type) for exc_type in error_types)
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Graceful 성능 저하로 함수 실행
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 결과 또는 폴백 결과
        
        Raises:
            Exception: 모든 폴백 실패 시
        """
        last_error = None
        
        try:
            # 원본 함수 실행
            logger.debug(f"[GracefulDegradation] 원본 함수 실행: {func.__name__}")
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        except Exception as e:
            last_error = e
            logger.error(
                f"[GracefulDegradation] 원본 함수 실패: "
                f"{type(e).__name__}: {e}"
            )
        
        # 폴백 체인 시도
        for i, config in enumerate(self.fallback_chain):
            if not config.enabled:
                continue
            
            if not self._should_use_fallback(last_error, config.error_types):
                logger.debug(
                    f"[GracefulDegradation] 폴백 {i+1}: 에러 타입 불일치"
                )
                continue
            
            try:
                logger.info(
                    f"[GracefulDegradation] 폴백 {i+1}/{len(self.fallback_chain)} 시도"
                )
                
                result = await config.strategy.execute(
                    last_error,
                    *args,
                    **kwargs
                )
                
                logger.info(
                    f"[GracefulDegradation] 폴백 {i+1} 성공"
                )
                return result
            
            except Exception as fallback_error:
                logger.warning(
                    f"[GracefulDegradation] 폴백 {i+1} 실패: {fallback_error}"
                )
                last_error = fallback_error
                continue
        
        # 모든 폴백 실패
        logger.error(
            f"[GracefulDegradation] 모든 폴백 실패"
        )
        raise last_error


class ServiceAvailability:
    """
    서비스 가용성 관리
    
    현재 시스템 상태에 따라 기능 제한
    """
    
    def __init__(self):
        self.available_services: Dict[str, bool] = {}
        self.degradation_level = 0  # 0 = 정상, 1 = 경미, 2 = 심각
    
    def mark_service_unavailable(self, service_name: str) -> None:
        """서비스 사용 불가 표시"""
        self.available_services[service_name] = False
        logger.warning(f"[Availability] {service_name} 사용 불가")
        self._update_degradation_level()
    
    def mark_service_available(self, service_name: str) -> None:
        """서비스 사용 가능 표시"""
        self.available_services[service_name] = True
        logger.info(f"[Availability] {service_name} 사용 가능")
        self._update_degradation_level()
    
    def is_service_available(self, service_name: str) -> bool:
        """서비스 가용성 확인"""
        return self.available_services.get(service_name, True)
    
    def _update_degradation_level(self) -> None:
        """저하 수준 업데이트"""
        unavailable_count = sum(
            1 for available in self.available_services.values()
            if not available
        )
        
        total_services = len(self.available_services)
        
        if total_services == 0:
            self.degradation_level = 0
        elif unavailable_count == 0:
            self.degradation_level = 0
        elif unavailable_count < total_services / 2:
            self.degradation_level = 1  # 경미
        else:
            self.degradation_level = 2  # 심각
        
        logger.debug(
            f"[Availability] 저하 수준: {self.degradation_level} "
            f"({unavailable_count}/{total_services})"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "available_services": self.available_services,
            "degradation_level": self.degradation_level,
            "is_degraded": self.degradation_level > 0,
        }


# 전역 인스턴스

_graceful_degradation: Optional[GracefulDegradation] = None
_service_availability: Optional[ServiceAvailability] = None


def get_graceful_degradation() -> GracefulDegradation:
    """Graceful Degradation 인스턴스"""
    global _graceful_degradation
    if _graceful_degradation is None:
        _graceful_degradation = GracefulDegradation()
    return _graceful_degradation


def get_service_availability() -> ServiceAvailability:
    """서비스 가용성 인스턴스"""
    global _service_availability
    if _service_availability is None:
        _service_availability = ServiceAvailability()
    return _service_availability


def reset_fallback_handler():
    """폴백 핸들러 리셋"""
    global _graceful_degradation, _service_availability
    _graceful_degradation = None
    _service_availability = None
