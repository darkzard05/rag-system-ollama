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
    # reset_or_add 리듀서 계약: 재생성 시 반드시 상수 델타 1을 반환해야 한다.
    # 키를 생략하면 리듀서가 0으로 리셋해 generate↔verify 무한 루프가 된다.
    assert result["regeneration_count"] == 1


@pytest.mark.asyncio
async def test_verify_exception_after_one_regeneration_ends(
    monkeypatch, state_cap_reached, config_with_failing_llm
):
    """재생성 1회 후 예외가 나도 카운터가 리셋되지 않고 end로 종료된다."""
    _enable_verification(monkeypatch)
    result = await verify_answer(
        state_cap_reached, config_with_failing_llm, writer=_NullWriter()
    )
    assert result["verification_route"] == "end"
    # end 경로는 카운터를 건드리지 않는다 (추가 재생성 없음).
    assert "regeneration_count" not in result


@pytest.mark.asyncio
async def test_verify_invalid_citation_increments_counter(monkeypatch):
    """환각 인용([doc:...])도 재생성 카운터를 1 증가시킨다."""
    _enable_verification(monkeypatch)
    state = {
        "response": "근거 [doc:deadbeefdeadbeef]",
        "relevant_docs": [],
        "input": "q",
        "regeneration_count": 0,
    }
    config = {"configurable": {"llm": MagicMock()}}
    result = await verify_answer(state, config, writer=_NullWriter())
    assert result["verification_route"] == "regenerate"
    assert result["regeneration_count"] == 1
    assert "유효하지 않은 인용" in result["verification_issues"][0]


@pytest.mark.asyncio
async def test_verify_invalid_citation_after_one_regeneration_ends(monkeypatch):
    """잘못된 인용이 반복돼도 재생성 1회 후에는 end로 종료된다."""
    _enable_verification(monkeypatch)
    state = {
        "response": "근거 [doc:deadbeefdeadbeef]",
        "relevant_docs": [],
        "input": "q",
        "regeneration_count": 1,
    }
    config = {"configurable": {"llm": MagicMock()}}
    result = await verify_answer(state, config, writer=_NullWriter())
    assert result["verification_route"] == "end"


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
