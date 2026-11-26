"""Bounded latency window stays O(1) per streamed chunk."""

from __future__ import annotations

from api.adaptive_controller import AdaptiveStreamingController


def test_latency_window_is_bounded_and_metrics_stay_correct() -> None:
    controller = AdaptiveStreamingController(client_profile="api")
    for i in range(controller.max_samples * 2):
        controller.record_latency(float(i))

    assert len(controller.latency_samples) == controller.max_samples
    metrics = controller.get_metrics()
    assert metrics["sample_count"] == controller.max_samples
    assert metrics["min_latency_ms"] == float(controller.max_samples)
    assert metrics["max_latency_ms"] == float(controller.max_samples * 2 - 1)
