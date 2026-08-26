import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph_builder import verify_answer


class _NullWriter:
    def __call__(self, data: object) -> None:
        pass


@pytest.fixture
def state_under_cap():
    return {
        "response": "답변",
        "relevant_docs": [],
        "input": "q",
        "regeneration_count": 0,
    }


@pytest.fixture
def state_cap_reached():
    return {
        "response": "답변",
        "relevant_docs": [],
        "input": "q",
        "regeneration_count": 1,
    }


@pytest.fixture
def config_with_failing_llm():
    cfg = {"configurable": {"llm": MagicMock()}}
    cfg["configurable"]["llm"].ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    return cfg


@pytest.fixture
def config_with_ok_llm():
    cfg = {"configurable": {"llm": MagicMock()}}
    cfg["configurable"]["llm"].ainvoke = AsyncMock(
        return_value=SimpleNamespace(content='{"faithful": true, "issues": []}')
    )
    return cfg


def _enable_verification(monkeypatch):
    monkeypatch.setattr("common.config.VERIFICATION_ENABLED", True)
    monkeypatch.setattr("common.config.VERIFICATION_SAMPLE_RATE", 1.0)
    # 샘플링 조기 리턴 방지
    monkeypatch.setattr(random, "random", lambda: 0.0)


@pytest.mark.asyncio
async def test_verify_exception_returns_regenerate_when_under_cap(
    monkeypatch, state_under_cap, config_with_failing_llm
):
    _enable_verification(monkeypatch)
    result = await verify_answer(
        state_under_cap, config_with_failing_llm, writer=_NullWriter()
    )
    assert result["verification_route"] == "regenerate"
    assert "검증 실행 오류" in result["verification_issues"][0]


@pytest.mark.asyncio
async def test_verify_exception_returns_end_when_cap_reached(
    monkeypatch, state_cap_reached, config_with_failing_llm
):
    _enable_verification(monkeypatch)
    result = await verify_answer(
        state_cap_reached, config_with_failing_llm, writer=_NullWriter()
    )
    assert result["verification_route"] == "end"


@pytest.mark.asyncio
async def test_verify_normal_pass_still_returns_end(
    monkeypatch, state_under_cap, config_with_ok_llm
):
    _enable_verification(monkeypatch)
    result = await verify_answer(
        state_under_cap, config_with_ok_llm, writer=_NullWriter()
    )
    assert result["verification_route"] == "end"
