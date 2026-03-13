import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from core.graph_builder import (
    preprocess, retrieve_and_rerank, grade_documents, rewrite_query, generate,
    GraphState, GradeResponse, RewriteResponse
)

@pytest.fixture
def mock_llm():
    """LLM과 구조화된 출력을 모킹합니다."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    
    # astream 모킹 (비동기 제너레이터)
    async def mock_astream(*args, **kwargs):
        chunk = MagicMock()
        chunk.content = "테스트 답변입니다."
        chunk.response_metadata = {"prompt_eval_count": 10}
        yield chunk

    llm.astream = mock_astream
    
    # CustomOllama의 전처리 메서드 모사
    def mock_convert(chunk):
        return chunk.content, None
    llm._convert_chunk_to_thought_and_content = mock_convert
    
    structured_llm = AsyncMock()
    llm.with_structured_output.return_value = structured_llm
    return llm, structured_llm

@pytest.fixture
def mock_retrievers():
    """BM25 및 FAISS 리트리버를 모킹합니다."""
    bm25 = AsyncMock()
    faiss = AsyncMock()
    return bm25, faiss

@pytest.fixture
def compiled_workflow():
    """실제 graph_builder.build_graph()와 동일한 구조의 테스트용 그래프를 생성합니다."""
    from core.graph_builder import build_graph
    return build_graph()

@pytest.mark.asyncio
async def test_workflow_general_intent(compiled_workflow, mock_llm):
    """일상 대화(General) 흐름 테스트: preprocess -> generate -> END"""
    llm, structured_llm = mock_llm
    
    config = {"configurable": {"llm": llm, "thread_id": "test_thread"}}
    inputs = {"input": "안녕", "chat_history": [], "retry_count": 0, "search_queries": []}
    
    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)
    
    # Verify
    assert result["intent"] == "general"
    assert "테스트 답변" in result["response"]

@pytest.mark.asyncio
async def test_workflow_rag_success_path(compiled_workflow, mock_llm, mock_retrievers):
    """RAG 성공 흐름 테스트: preprocess -> retrieve -> grade(YES) -> generate -> END"""
    llm, structured_llm = mock_llm
    bm25, faiss = mock_retrievers
    
    # Mock Retrieve
    doc = Document(page_content="RAG 지식", metadata={"page": 1})
    bm25.ainvoke.return_value = [doc]
    faiss.ainvoke.return_value = []
    
    # Mock Grade
    structured_llm.ainvoke.return_value = GradeResponse(
        is_relevant=True, 
        reason="Matched",
        relevant_entities=["RAG"]
    )
    
    config = {
        "configurable": {
            "llm": llm,
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "thread_id": "test_thread_rag"
        }
    }
    inputs = {"input": "RAG가 뭐야?", "chat_history": []}
    
    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)
    
    # Verify
    assert result["intent"] == "generate" # grade 이후 전이됨
    assert len(result["relevant_docs"]) > 0
    assert "테스트 답변" in result["response"]

@pytest.mark.asyncio
async def test_workflow_rag_retry_path(compiled_workflow, mock_llm, mock_retrievers):
    """RAG 재시도 흐름 테스트"""
    llm, structured_llm = mock_llm
    bm25, faiss = mock_retrievers
    
    bm25.ainvoke.side_effect = [[Document(page_content="무관한 정보")], [Document(page_content="정확한 정보")]]
    faiss.ainvoke.return_value = []
    
    structured_llm.ainvoke.side_effect = [
        GradeResponse(is_relevant=False, reason="Irrelevant", relevant_entities=[]), # 1차 Grade
        RewriteResponse(optimized_query="재구성 쿼리"), # Rewrite
        GradeResponse(is_relevant=True, reason="Matched", relevant_entities=["정보"]), # 2차 Grade
    ]
    
    config = {
        "configurable": {
            "llm": llm,
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "thread_id": "test_thread_rag"
        }
    }
    inputs = {"input": "정보를 알려줘", "chat_history": []}
    
    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)
    
    # Verify
    assert result["retry_count"] == 1
    assert "재구성 쿼리" in result["search_queries"]
    assert "테스트 답변" in result["response"]

@pytest.mark.asyncio
async def test_workflow_cache_hit_path(compiled_workflow, mock_llm):
    """캐시 적중 시 즉시 종료 테스트"""
    llm, structured_llm = mock_llm
    
    with patch("core.graph_builder.get_response_cache") as mock_cache_factory:
        mock_cache = AsyncMock()
        # cached_res.response 속성을 가진 객체 모사
        mock_res = MagicMock()
        mock_res.response = "캐시된 결과"
        mock_res.metadata = {"thought": "캐시된 생각"}
        
        mock_cache.get.return_value = mock_res
        mock_cache_factory.return_value = mock_cache

        config = {"configurable": {"llm": llm, "thread_id": "test_thread_cache"}}
        inputs = {"input": "테스트", "chat_history": []}
        
        # Execute
        result = await compiled_workflow.ainvoke(inputs, config=config)
        
        # Verify
        assert result["is_cached"] is True
        assert result["response"] == "캐시된 결과"
