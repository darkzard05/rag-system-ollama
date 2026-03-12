import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from core.graph_builder import grade_documents, rewrite_query, GradeResponse, RewriteResponse

@pytest.mark.asyncio
async def test_grade_documents_relevant():
    # Setup
    llm = MagicMock()
    # with_structured_output이 호출될 때 반환할 mock 객체 설정
    structured_llm = AsyncMock()
    structured_llm.ainvoke.return_value = GradeResponse(
        is_relevant=True, 
        reason="Direct answer found",
        relevant_entities=["weather"]
    )
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "How is the weather?",
        "relevant_docs": [Document(page_content="The weather is sunny.")]
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer)

    # Verify
    assert result["intent"] == "generate"
    llm.with_structured_output.assert_called_with(GradeResponse)

@pytest.mark.asyncio
async def test_grade_documents_irrelevant():
    # Setup
    llm = MagicMock()
    structured_llm = AsyncMock()
    structured_llm.ainvoke.return_value = GradeResponse(
        is_relevant=False, 
        reason="No mention of weather",
        relevant_entities=[]
    )
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "How is the weather?",
        "relevant_docs": [Document(page_content="I like pizza.")]
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer)

    # Verify
    assert result["intent"] == "transform"

@pytest.mark.asyncio
async def test_grade_documents_ambiguous_term_case():
    """cm3 vs CM3 모델과 같은 중의적 용어 상황 테스트"""
    # Setup
    llm = MagicMock()
    structured_llm = AsyncMock()
    # 개선된 프롬프트 덕분에 'cm3'를 모델명으로 인식했다고 가정
    structured_llm.ainvoke.return_value = GradeResponse(
        is_relevant=True, 
        reason="Document describes 'CM3' as a multimodal model name, matching user query 'cm3'.",
        relevant_entities=["CM3", "multimodal model"]
    )
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "cm3가 뭔가요?",
        "relevant_docs": [Document(page_content="CM3 is a causally-masked multimodal model.")]
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer)

    # Verify
    assert result["intent"] == "generate"
    # 실제 invoke 시 전달된 프롬프트에 '판단 원칙' 등이 포함되었는지는 수동 검증 또는 프롬프트 스냅샷 테스트 필요

@pytest.mark.asyncio
async def test_rewrite_query_success():
    # Setup
    llm = MagicMock()
    structured_llm = AsyncMock()
    structured_llm.ainvoke.return_value = RewriteResponse(optimized_query="current weather forecast")
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "weather?",
        "retry_count": 0,
        "search_queries": []
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await rewrite_query(state, config, writer)

    # Verify
    assert "current weather forecast" in result["search_queries"]
    assert result["retry_count"] == 1
    llm.with_structured_output.assert_called_with(RewriteResponse)

@pytest.mark.asyncio
async def test_structured_output_failure_fallback():
    # Setup - LLM이 에러를 던질 때
    llm = MagicMock()
    structured_llm = AsyncMock()
    structured_llm.ainvoke.side_effect = Exception("JSON Parsing Error")
    llm.with_structured_output.return_value = structured_llm

    state = {
        "input": "test",
        "relevant_docs": [Document(page_content="test")]
    }
    config = {"configurable": {"llm": llm}}
    writer = MagicMock()

    # Execute
    result = await grade_documents(state, config, writer)

    # Verify - Fallback (Exception handling)
    assert result["intent"] == "generate" # Default fallback in code
