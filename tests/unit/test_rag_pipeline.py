import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from common.exceptions import EmptyPDFError, VectorStoreError
from core.rag_core import RAGSystem

@pytest.fixture
def rag_system():
    return RAGSystem(session_id="test_session")

@pytest.mark.asyncio
async def test_build_pipeline_success(rag_system):
    """정상적인 파이프라인 구축 흐름 테스트"""
    mock_docs = [Document(page_content="테스트 본문", metadata={"page": 1})]
    mock_splits = [Document(page_content="청크 1")]
    mock_vectors = [[0.1, 0.2]]
    
    # 의존성 모킹
    with patch("core.rag_core.load_pdf_docs", return_value=mock_docs) as mock_load, \
         patch("core.rag_core.split_documents", AsyncMock(return_value=(mock_splits, mock_vectors))) as mock_split, \
         patch("core.rag_core.create_vector_store") as mock_vs, \
         patch("core.rag_core.create_bm25_retriever") as mock_bm25, \
         patch("core.rag_core.compute_file_hash", return_value="hash123"), \
         patch("core.rag_core.get_resource_pool") as mock_pool, \
         patch("core.rag_core.SessionManager") as mock_session:
        
        mock_pool_instance = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        embedder = MagicMock()
        
        # Execute
        msg, is_cached = await rag_system.build_pipeline(
            file_path="dummy.pdf", 
            file_name="dummy.pdf", 
            embedder=embedder
        )
        
        # Verify
        assert "신규 인덱싱 완료" in msg
        assert is_cached is False
        mock_load.assert_called_once()
        mock_split.assert_called_once()
        mock_vs.assert_called_once()
        mock_bm25.assert_called_once()
        mock_pool_instance.register.assert_called_once()

@pytest.mark.asyncio
async def test_build_pipeline_empty_pdf(rag_system):
    """빈 PDF 입력 시 에러 발생 테스트"""
    with patch("core.rag_core.load_pdf_docs", return_value=[]), \
         patch("core.rag_core.compute_file_hash", return_value="hash123"), \
         patch("core.rag_core.SessionManager"):
        
        embedder = MagicMock()
        
        with pytest.raises(EmptyPDFError):
            await rag_system.build_pipeline("empty.pdf", "empty.pdf", embedder)

@pytest.mark.asyncio
async def test_aquery_success(rag_system):
    """정상적인 질의 응답 흐방 테스트"""
    mock_engine = AsyncMock()
    mock_engine.ainvoke.return_value = {
        "response": "답변입니다.",
        "thought": "생각 중...",
        "relevant_docs": [Document(page_content="근거", metadata={"page": 1})],
        "performance": {}
    }
    
    with patch("core.rag_core.SessionManager") as mock_session, \
         patch.object(rag_system, "_prepare_config", AsyncMock(return_value={})), \
         patch.object(rag_system, "_hydrate_docs") as mock_hydrate:
        
        mock_session.get.return_value = mock_engine
        
        # Execute
        result = await rag_system.aquery("질문")
        
        # Verify
        assert result["response"] == "답변입니다."
        mock_engine.ainvoke.assert_called_once()
        mock_hydrate.assert_called_once()

@pytest.mark.asyncio
async def test_aquery_not_ready(rag_system):
    """파이프라인 구축 전 질의 시 에러 발생 테스트"""
    with patch("core.rag_core.SessionManager") as mock_session:
        mock_session.get.return_value = None # rag_engine 없음
        
        with pytest.raises(VectorStoreError):
            await rag_system.aquery("질문")
