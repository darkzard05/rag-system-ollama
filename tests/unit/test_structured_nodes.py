import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.graph_builder import grade_documents


def json_response(**fields):
    """LLM JSON 모드 응답을 모사하는 SimpleNamespace 객체를 생성합니다."""
    return SimpleNamespace(content=json.dumps(fields))


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_grade_documents_relevant(mock_dispatch):
    # Setup
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        reason="Direct answer found",
        relevant_entities=["weather"],
        optimized_query=None,
    )

    state = {
        "input": "How is the weather?",
        "relevant_docs": [Document(page_content="The weather is sunny.")],
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer=writer)

    # Verify
    assert result["intent"] == "generate"
    llm.bind.assert_called_once()
    json_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_grade_documents_irrelevant(mock_dispatch):
    # Setup
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        reason="No mention of weather",
        relevant_entities=[],
        optimized_query="current weather forecast",
    )

    state = {
        "input": "How is the weather?",
        "relevant_docs": [Document(page_content="I like pizza.")],
        "retry_count": 0,
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer=writer)

    # Verify
    assert result["intent"] == "transform"
    assert result["search_queries"] == ["current weather forecast"]
    assert result["retry_count"] == 1


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_grade_documents_ambiguous_term_case(mock_dispatch):
    """cm3 vs CM3 모델과 같은 중의적 용어 상황 테스트"""
    # Setup
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    # 개선된 프롬프트 덕분에 'cm3'를 모델명으로 인식했다고 가정
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        reason=(
            "Document describes 'CM3' as a multimodal model name, "
            "matching user query 'cm3'."
        ),
        relevant_entities=["CM3", "multimodal model"],
        optimized_query=None,
    )

    state = {
        "input": "cm3가 뭔가요?",
        "relevant_docs": [
            Document(page_content="CM3 is a causally-masked multimodal model.")
        ],
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer=writer)

    # Verify
    assert result["intent"] == "generate"
    # 실제 invoke 시 전달된 프롬프트에 '판단 원칙' 등이 포함되었는지는 수동 검증 또는 프롬프트 스냅샷 테스트 필요


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_grade_documents_rewrite_integration(mock_dispatch):
    """rewrite는 별도 노드 호출이 아닌 grade_documents의 rewrite 경로로 통합 검증합니다."""
    # Setup
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        reason="No relevant information",
        relevant_entities=[],
        optimized_query="current weather forecast",
    )

    state = {
        "input": "weather?",
        "retry_count": 0,
        "search_queries": [],
        "relevant_docs": [Document(page_content="irrelevant content")],
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer=writer)

    # Verify
    assert result["intent"] == "transform"
    assert result["search_queries"] == ["current weather forecast"]
    assert result["retry_count"] == 1


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_structured_output_failure_fallback(mock_dispatch):
    """JSON 모드와 수동 파싱 폴백 경로 모두 실패 시 기본값(transform)으로 폴백합니다."""
    # Setup - LLM이 에러를 던질 때 (JSON 모드 + 폴백 경로 모두 ValueError)
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.side_effect = ValueError("JSON Parsing Error")
    llm.ainvoke.side_effect = ValueError("Fallback Error")

    state = {
        "input": "test",
        "relevant_docs": [Document(page_content="test")],
        "retry_count": 0,
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer=writer)

    # Verify - Fallback (Exception handling)
    assert result["intent"] == "transform"
    assert result["retry_count"] == 1
