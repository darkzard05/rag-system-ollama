"""GraphState 턴 경계 리셋 리듀서(reset_or_append / reset_or_add) 및 preprocess 리셋 동작 검증.

버그 배경: search_queries / retry_count가 operator.add 리듀서로 체크포인터에 턴 간 누적되어
retrieve_and_rerank가 이전 턴의 stale 재작성 쿼리를 사용했다. 새 리듀서는 턴 시작 리셋 신호
(빈 리스트 / 0)를 받으면 기존 누적 값을 비운다.
"""

from api.schemas import reset_or_add, reset_or_append
from core.graph_builder import preprocess


class MockWriter:
    """preprocess의 StreamWriter 위치 인자를 대체하는 모의 객체."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


def test_reset_or_append_empty_clears_stale_queries():
    """턴 시작 리셋: 새 값이 빈 리스트면 이전 턴의 재작성 쿼리를 비운다."""
    assert reset_or_append(["stale"], []) == []


def test_reset_or_append_first_query():
    """빈 기존 리스트에 첫 재작성 쿼리를 append한다."""
    assert reset_or_append([], ["r"]) == ["r"]


def test_reset_or_append_accumulates_within_turn():
    """동일 턴 내 재작성 쿼리를 누적 append한다."""
    assert reset_or_append(["r1"], ["r2"]) == ["r1", "r2"]


def test_reset_or_add_zero_resets_retry_budget():
    """턴 시작 리셋: 새 값이 0이면 이전 턴의 재시도 횟수를 0으로 리셋한다."""
    assert reset_or_add(1, 0) == 0


def test_reset_or_add_first_retry():
    """0에서 첫 재시도(+1)를 누적한다."""
    assert reset_or_add(0, 1) == 1


def test_reset_or_add_accumulates_within_turn():
    """동일 턴 내 재시도 횟수를 누적한다."""
    assert reset_or_add(1, 1) == 2


async def test_preprocess_resets_search_queries():
    """preprocess가 search_queries=[]를 반환해 턴 시작 리셋 신호를 보낸다."""
    state = {"input": "DeepSeek-R1 성능은 어때?", "chat_history": []}
    config = {"configurable": {}}
    result = await preprocess(state, config, writer=MockWriter())
    assert result["search_queries"] == []
    assert result["retry_count"] == 0
