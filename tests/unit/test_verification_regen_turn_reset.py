"""A verification retry budget must reset at the start of every turn."""

from core.graph._grade_verify import _verification_retry
from core.graph.graph_builder import preprocess


class MockWriter:
    """Stand-in for the LangGraph stream writer."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


async def test_next_turn_can_regenerate_after_a_prior_failure() -> None:
    """preprocess clears the retry budget so a new turn can self-correct."""
    prior_turn_end = {"input": "이전 질문", "regeneration_count": 1}

    next_turn_out = await preprocess(
        {"input": "새 질문입니다"}, {"configurable": {}}, writer=MockWriter()
    )
    merged = {**prior_turn_end, **next_turn_out}

    assert next_turn_out.get("regeneration_count") == 0
    decision = _verification_retry(merged, ["유효하지 않은 인용: [doc:x]"])
    assert decision["verification_route"] == "regenerate"
