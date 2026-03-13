import pytest
from unittest.mock import AsyncMock, MagicMock
from core.graph_builder import rewrite_query, preprocess, RewriteResponse

@pytest.mark.asyncio
async def test_rewrite_query_reducer_logic():
    # Setup
    llm = MagicMock()
    structured_llm = AsyncMock()
    structured_llm.ainvoke.return_value = RewriteResponse(optimized_query="new search term")
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "original query",
        "retry_count": 0,
        "search_queries": ["existing query"]
    }
    # config에 llm 주입
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    update = await rewrite_query(state, config, writer)

    # Verify
    # 리듀서가 적용된 그래프에서는 이 리턴값이 기존 상태와 합쳐지지만, 
    # 단위 테스트(노드 함수 직접 호출)에서는 리턴되는 델타값만 확인합니다.
    assert update["search_queries"] == ["new search term"]
    assert update["retry_count"] == 1

@pytest.mark.asyncio
async def test_preprocess_initial_state():
    state = {"input": "안녕"}
    config = {}
    writer = MagicMock()

    update = await preprocess(state, config, writer)

    assert update["intent"] == "general"
    # 리듀서 필드(retry_count, search_queries)는 초기화 단계에서 명시적으로 0이나 []를 줄 필요가 없음 
    # (이미 State 정의에서 Annotated/operator.add로 되어있어 그래프 시작 시 기본값이 적용됨)
    assert "retry_count" not in update
