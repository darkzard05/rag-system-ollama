import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from api.schemas import GraphState

from core.graph_builder import grade_documents


def json_response(**fields):
    """LLM JSON 모드 응답을 모사하는 SimpleNamespace 객체를 생성합니다."""
    return SimpleNamespace(content=json.dumps(fields))


@pytest.fixture
def mock_state():
    return {
        "input": "DeepSeek-R1의 성능은 어떠한가요?",
        "relevant_docs": [
            Document(
                page_content="DeepSeek-R1은 매우 높은 성능을 가진 모델입니다.",
                metadata={"rerank_score": 0.5},
            ),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }


@pytest.fixture
def mock_config():
    """bind+JSON 모드로 모킹된 LLM을 포함한 config를 생성합니다."""
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    return {"configurable": {"llm": llm}}


@pytest.mark.asyncio
async def test_grade_documents_short_circuit():
    """리랭킹 점수가 매우 높을 때 Short-circuit이 작동하여 즉시 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 최고", metadata={"rerank_score": 0.9}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    llm = MagicMock()
    config = {"configurable": {"llm": llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {"intent": "generate"}
    # Short-circuit에서는 LLM을 호출하지 않아야 합니다.
    llm.ainvoke.assert_not_called()
    llm.bind.assert_not_called()


@pytest.mark.asyncio
async def test_grade_documents_rubric_pass():
    """Unified 응답(action=generate)이면 LLM 검증을 통과하여 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 보통", metadata={"rerank_score": 0.5}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        relevant_entities=["DeepSeek-R1"],
        reason="충분한 정보 포함",
        optimized_query=None,
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {"intent": "generate"}


@pytest.mark.asyncio
async def test_grade_documents_rubric_fail():
    """Unified 응답(action=rewrite)이면 재작성 쿼리와 함께 transform을 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 낮음", metadata={"rerank_score": 0.3}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        relevant_entities=["DeepSeek-R1"],
        reason="정보는 있으나 부족함",
        optimized_query="DeepSeek-R1 성능 벤치마크",
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {
        "intent": "transform",
        "search_queries": ["DeepSeek-R1 성능 벤치마크"],
        "retry_count": 1,
    }


@pytest.mark.asyncio
async def test_grade_documents_is_relevant_false():
    """is_relevant가 False일 때 점수와 상관없이 transform을 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="전혀 상관없는 내용", metadata={"rerank_score": 0.5}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        relevant_entities=[],
        reason="관련 없음",
        optimized_query="DeepSeek-R1 리뷰",
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {
        "intent": "transform",
        "search_queries": ["DeepSeek-R1 리뷰"],
        "retry_count": 1,
    }
