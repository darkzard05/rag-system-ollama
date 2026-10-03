"""first_token_ms / ttft_ms 분리 회귀 테스트.

``models.thinking: true``일 때 Ollama은 추론 블록을 먼저 토큰화한다. core의
``ttft_ms``는 첫 **content** 청크를 재므로 thinking 생성 시간이 TTFT에 흡수된다
(실측 2.2~3.9초). 이번 수치는 thinking 비용과 실사용자 대기를 분리하기 위한
**추가(additive)** 지표다 — ``ttft_ms``의 기존 의미는 바꾸지 않는다.

계약:
- ``first_token_ms``: 첫 출력(thought 또는 content) — API의 ``first_token_latency``와 정렬
- ``ttft_ms``: 첫 content — 기존 값 유지 (bench_query_latency.py 회귀 게이트 보존)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import core.graph._generate as gen


class _FakeChunk:
    """astream이 머무는 최소 청크 객체."""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.response_metadata = metadata or {}


def _make_llm(pairs: list[tuple[str, str]]) -> MagicMock:
    """(thought, content) 시퀀스를 가진 astream LLM 더블을 만든다."""
    chunks = [_FakeChunk(content=c) for _, c in pairs]

    async def _astream(_messages: Any, config: Any = None) -> Any:
        for c in chunks:
            yield c

    llm = MagicMock()
    llm.bind.return_value.astream = _astream
    state = {"i": 0}

    def _convert(_chunk: Any) -> tuple[str, str]:
        thought, content = pairs[state["i"]]
        state["i"] += 1
        return content, thought

    llm._convert_chunk_to_thought_and_content = _convert
    return llm


@patch.object(gen, "get_resource_manager")
@patch.object(gen.ModelManager, "inference_session")
@pytest.mark.asyncio
async def test_thought_first_splits_thinking_from_ttft(
    inference_session: MagicMock, resource_manager: MagicMock
) -> None:
    """thought가 먼저 오면 first_token_ms < ttft_ms가 되어야 한다."""
    inference_session.return_value.__aenter__ = _aenter
    inference_session.return_value.__aexit__ = _aexit
    coordinator = resource_manager.return_value
    coordinator.use_llm.return_value.__aenter__ = _aenter
    coordinator.use_llm.return_value.__aexit__ = _aexit

    # t0=0.0 pass_start, t1=0.15 thought, t2=3.20 content, t3=4.00 end
    ticks = iter([0.0, 0.15, 3.20, 4.00])
    llm = _make_llm([("추론 중...", ""), ("", '{"final_answer":"답변"}')])

    with patch.object(gen.time, "perf_counter", lambda: next(ticks)):
        result = await gen._generate_pass(
            llm=llm,
            sys_msg=MagicMock(),
            human_msg=MagicMock(),
            prompt_version="v1",
            config={},
            writer=None,
            attempt=1,
        )

    assert result["first_token_ms"] == pytest.approx(150.0)
    assert result["ttft_ms"] == pytest.approx(3200.0)
    # 핵심 계약: thinking 시간이 두 지표를 분리한다.
    assert result["first_token_ms"] < result["ttft_ms"]
    assert result["parse_failed"] is False


@patch.object(gen, "get_resource_manager")
@patch.object(gen.ModelManager, "inference_session")
@pytest.mark.asyncio
async def test_content_only_keeps_both_metrics_equal(
    inference_session: MagicMock, resource_manager: MagicMock
) -> None:
    """thinking이 없으면 두 지표가 같아야 한다 (회귀: 과도한 분리 금지)."""
    inference_session.return_value.__aenter__ = _aenter
    inference_session.return_value.__aexit__ = _aexit
    coordinator = resource_manager.return_value
    coordinator.use_llm.return_value.__aenter__ = _aenter
    coordinator.use_llm.return_value.__aexit__ = _aexit

    ticks = iter([0.0, 0.5, 0.5, 2.00])
    llm = _make_llm([("", '{"final_answer":"답변"}')])

    with patch.object(gen.time, "perf_counter", lambda: next(ticks)):
        result = await gen._generate_pass(
            llm=llm,
            sys_msg=MagicMock(),
            human_msg=MagicMock(),
            prompt_version="v1",
            config={},
            writer=None,
            attempt=1,
        )

    assert result["first_token_ms"] == pytest.approx(500.0)
    assert result["ttft_ms"] == pytest.approx(500.0)


async def _aenter(*args: Any, **kwargs: Any) -> Any:
    return None


async def _aexit(*args: Any, **kwargs: Any) -> Any:
    return False
