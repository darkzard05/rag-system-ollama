from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.graph.graph_builder import verify_answer


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
    # D19: 샘플링은 _verify 모듈 레벨 상수 기반 결정적 해시(_should_verify)로
    # 전환됨 — rate=1.0이면 항상 검증하므로 random 패치는 더 이상 필요 없다.
    monkeypatch.setattr("core.graph._verify.VERIFICATION_ENABLED", True)
    monkeypatch.setattr("core.graph._verify.VERIFICATION_SAMPLE_RATE", 1.0)


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
