"""
LLM 제공자별로 서로 다른 성능 메타데이터 형식을 표준 PerformanceStats 스키마로 변환하는 어댑터입니다.
"""

from typing import Any

from src.common.schemas import PerformanceStats


def adapt_llm_metadata(
    metadata: dict[str, Any], current_stats: PerformanceStats | None = None
) -> PerformanceStats:
    """
    LLM의 raw 메타데이터를 표준 PerformanceStats 객체로 변환합니다.

    Args:
        metadata: LLM 제공자가 반환한 raw 메타데이터 (e.g., Ollama response_metadata)
        current_stats: 기존에 계산된 통계 (스트리밍 중 계산된 시간 등 유지 목적)

    Returns:
        표준화된 PerformanceStats 객체
    """
    stats = current_stats or PerformanceStats()

    # 1. Ollama 메타데이터 매핑
    # Ollama는 prompt_eval_count (input)와 eval_count (output)를 사용합니다.
    if "prompt_eval_count" in metadata or "eval_count" in metadata:
        stats.input_token_count = metadata.get(
            "prompt_eval_count", stats.input_token_count
        )
        stats.token_count = metadata.get("eval_count", stats.token_count)

    # 2. 일반적인 token_count/input_token_count 매핑 (Fallback)
    elif "token_count" in metadata or "input_token_count" in metadata:
        stats.token_count = metadata.get("token_count", stats.token_count)
        stats.input_token_count = metadata.get(
            "input_token_count", stats.input_token_count
        )

    # 3. 기타 메타데이터 처리 (필요 시 추가)
    # 예: total_duration 등을 통한 시간 보정
    if "total_duration" in metadata:
        # Ollama's total_duration is in nanoseconds
        duration_sec = metadata["total_duration"] / 1e9
        stats.total_time = duration_sec

    return stats
