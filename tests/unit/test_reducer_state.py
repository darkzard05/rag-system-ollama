import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.graph_builder import rewrite_query, preprocess


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_rewrite_query_reducer_logic(mock_dispatch):
    """search_queries가 이미 있으면 rewrite_query는 순수 passthrough(빈 델타)를 반환합니다.

    retry_count는 grade_documents가 이미 +1을 적용했으므로 여기서 추가 증가하면
    operator.add 리듀서 합산으로 재시도 예산이 이중 소진됩니다.
    """
    # Setup
    llm = MagicMock()

    state = {
        "input": "original query",
        "retry_count": 1,
        "search_queries": ["existing query"],
    }
    # config에 llm 주입
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    update = await rewrite_query(state, config, writer=writer)

    # Verify
    # 리듀서가 적용된 그래프에서는 빈 델타가 기존 상태를 그대로 유지시킵니다.
    assert update == {}
    llm.ainvoke.assert_not_called()
    llm.bind.assert_not_called()


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_rewrite_query_fallback_increments_retry(mock_dispatch):
    """search_queries가 없으면 폴백으로 retry_count만 1 증가시킵니다."""
    # Setup
    llm = MagicMock()

    state = {"input": "original query", "retry_count": 0, "search_queries": []}
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    update = await rewrite_query(state, config, writer=writer)

    # Verify
    assert update == {"retry_count": 1}
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_preprocess_initial_state():
    state = {"input": "안녕"}
    config = {}
    writer = MagicMock()

    update = await preprocess(state, config, writer=writer)

    assert update["intent"] == "general"
    # preprocess는 시작 노드이므로 이후 노드들이 operator.add 리듀서로 합산할 때
    # 기준점이 되도록 retry_count를 명시적으로 0으로 초기화합니다.
    # (search_queries는 이 시점에서 아직 생성되지 않음)
    assert update["retry_count"] == 0
