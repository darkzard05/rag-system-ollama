"""
적응형 스트리밍 컨트롤러 모듈 - Phase 2.3 버퍼 크기 최적화
Task 12의 streaming_handler.py에서 분리 (PHASE 3-P1)
"""

import logging
from typing import cast

from common.config import UI_CONTENT_BUFFER_SIZE

logger = logging.getLogger(__name__)

__all__ = ["AdaptiveStreamingController", "get_adaptive_controller"]


class AdaptiveStreamingController:
    """
    적응형 스트리밍 컨트롤러 v2 - Phase 2.3
    - EWMA(Exponentially Weighted Moving Average) 지연 추정
    - 클라이언트 프로파일(streamlit/api/websocket)별 최적화
    """

    # 클라이언트 프로파일 설정
    PROFILES = {
        "streamlit": {"alpha": 0.3, "target_latency_ms": 50, "priority": "low_latency"},
        "api": {"alpha": 0.2, "target_latency_ms": 200, "priority": "throughput"},
        "websocket": {"alpha": 0.25, "target_latency_ms": 100, "priority": "balanced"},
    }

    def __init__(
        self,
        initial_buffer_size: int = 1,
        min_buffer_size: int = 1,
        max_buffer_size: int = 10,
        client_profile: str = "streamlit",
    ):
        self.current_buffer_size = initial_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.client_profile = client_profile
        self.profile_config = self.PROFILES.get(
            client_profile, self.PROFILES["streamlit"]
        )

        # EWMA 상태
        self.ewma_latency: float | None = None
        self.alpha = cast(float, self.profile_config["alpha"])
        self.target_latency_ms = cast(float, self.profile_config["target_latency_ms"])

        # 메트릭
        self.latency_samples: list[float] = []
        self.max_samples = 50
        self.buffer_adjustments = 0
        self.profile_switches = 0

    def record_latency(self, latency_ms: float) -> None:
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_samples:
            self.latency_samples.pop(0)

        # EWMA 업데이트
        if self.ewma_latency is None:
            self.ewma_latency = latency_ms
        else:
            self.ewma_latency = (
                self.alpha * latency_ms + (1 - self.alpha) * self.ewma_latency
            )

        self._adjust_buffer_size()

    def _adjust_buffer_size(self) -> None:
        if self.ewma_latency is None:
            return

        target = self.target_latency_ms
        ewma = self.ewma_latency

        # 목표 지연 대비 버퍼 크기 조정
        if ewma > target * 1.5:  # 지연이 목표의 1.5배 초과
            new_size = min(self.current_buffer_size + 1, self.max_buffer_size)
            if new_size != self.current_buffer_size:
                self.current_buffer_size = new_size
                self.buffer_adjustments += 1
        elif ewma < target * 0.7:  # 지연이 목표의 70% 미만
            new_size = max(self.current_buffer_size - 1, self.min_buffer_size)
            if new_size != self.current_buffer_size:
                self.current_buffer_size = new_size
                self.buffer_adjustments += 1

    def set_client_profile(self, profile: str) -> None:
        """클라이언트 프로파일 동적 전환"""
        if profile in self.PROFILES and profile != self.client_profile:
            old_profile = self.client_profile
            self.client_profile = profile
            self.profile_config = self.PROFILES[profile]
            self.alpha = cast(float, self.profile_config["alpha"])
            self.target_latency_ms = cast(
                float, self.profile_config["target_latency_ms"]
            )
            self.ewma_latency = None  # 리셋
            self.profile_switches += 1
            logger.info(
                f"[AdaptiveStreaming] 프로파일 변경: {old_profile} -> {profile}"
            )

    def get_buffer_size(self) -> int:
        return self.current_buffer_size

    def get_metrics(self) -> dict[str, object]:
        metrics = {
            "current_buffer_size": self.current_buffer_size,
            "client_profile": self.client_profile,
            "target_latency_ms": self.target_latency_ms,
            "ewma_latency_ms": self.ewma_latency or 0.0,
            "buffer_adjustments": self.buffer_adjustments,
            "profile_switches": self.profile_switches,
            "sample_count": len(self.latency_samples),
        }
        if self.latency_samples:
            metrics.update(
                {
                    "avg_latency_ms": sum(self.latency_samples)
                    / len(self.latency_samples),
                    "min_latency_ms": min(self.latency_samples),
                    "max_latency_ms": max(self.latency_samples),
                }
            )
        return metrics


def get_adaptive_controller(
    client_profile: str = "streamlit",
) -> AdaptiveStreamingController:
    return AdaptiveStreamingController(
        client_profile=client_profile, initial_buffer_size=UI_CONTENT_BUFFER_SIZE
    )
