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

    with (
        patch(
            "core.pipeline_builder.load_pdf_docs", return_value=mock_docs
        ) as mock_load,
        patch(
            "core.pipeline_builder.split_documents",
            AsyncMock(return_value=(mock_splits, mock_vectors)),
        ) as mock_split,
        patch("core.pipeline_builder.create_vector_store") as mock_vs,
        patch("core.pipeline_builder.create_bm25_retriever") as mock_bm25,
        patch("core.pipeline_builder.compute_file_hash", return_value="hash123"),
        patch("core.pipeline_builder.get_resource_manager") as mock_manager,
        patch("core.pipeline_builder.SessionManager") as mock_session,
        patch("core.pipeline_builder.ENABLE_VECTOR_CACHE", False),
        patch("core.pipeline_builder.build_graph", new_callable=AsyncMock),
        patch("core.pipeline_builder.VectorStoreCache") as mock_cache_cls,
    ):
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        mock_manager_instance.register_retrievers = AsyncMock()

        embedder = MagicMock()
        embedder.model = "test-model"

        msg, is_cached = await rag_system.build_pipeline(
            file_path="dummy.pdf",
            file_name="dummy.pdf",
            embedder=embedder,
        )

        assert "신규 인덱싱 완료" in msg
        assert is_cached is False
        mock_load.assert_called_once()
        mock_split.assert_called_once()
        mock_vs.assert_called_once()
        mock_bm25.assert_called_once()
        mock_manager_instance.register_retrievers.assert_called_once()


@pytest.mark.asyncio
async def test_build_pipeline_degenerate_no_token_corpus_vector_fallback(rag_system):
    """비-빈 청크지만 빈-토큰 코퍼스는 BM25 0나눗셈 없이 벡터 전용 폴백으로 성공해야 한다.

    회귀 배경: 파싱·청킹은 통과하지만 bm25_tokenizer로 빈 토큰을 만드는 코퍼스는
    rank_bm25._calc_idf가 분모 0(ZeroDivisionError)으로 크래시했다. 이 테스트는
    create_bm25_retriever를 실제(미패치)로 두고 빈-토큰 split을 흘려보내
    가드가 None(벡터 전용) 폴백으로 파이프라인을 정상 완료시키는지 검증한다.
    """
    mock_splits = [
        Document(page_content="")
    ]  # 비-빈 split → InsufficientChunksError 통과
    mock_vectors = [[0.1, 0.2]]

    with (
        patch(
            "core.pipeline_builder.load_pdf_docs",
            return_value=[Document(page_content="")],
        ),
        patch(
            "core.pipeline_builder.split_documents",
            AsyncMock(return_value=(mock_splits, mock_vectors)),
        ),
        patch("core.pipeline_builder.create_vector_store") as mock_vs,
        # create_bm25_retriever는 deliberately 패치하지 않음 → 실제 가드 실행
        patch("core.pipeline_builder.compute_file_hash", return_value="hash123"),
        patch("core.pipeline_builder.get_resource_manager") as mock_manager,
        patch("core.pipeline_builder.SessionManager"),
        patch("core.pipeline_builder.ENABLE_VECTOR_CACHE", False),
        patch("core.pipeline_builder.build_graph", new_callable=AsyncMock),
        patch("core.pipeline_builder.VectorStoreCache"),
    ):
        mock_manager.return_value.register_retrievers = AsyncMock()
        embedder = MagicMock()
        embedder.model = "test-model"

        msg, is_cached = await rag_system.build_pipeline(
            file_path="dummy.pdf", file_name="dummy.pdf", embedder=embedder
        )

        assert "신규 인덱싱 완료" in msg  # ZeroDivisionError 없이 성공
        assert is_cached is False
        mock_vs.assert_called_once()


@pytest.mark.asyncio
async def test_build_pipeline_empty_pdf(rag_system):
    """빈 PDF 입력 시 에러 발생 테스트"""
    with (
        patch("core.pipeline_builder.load_pdf_docs", return_value=[]),
        patch("core.pipeline_builder.compute_file_hash", return_value="hash123"),
        patch("core.pipeline_builder.SessionManager"),
        patch("core.pipeline_builder.ENABLE_VECTOR_CACHE", False),
        patch("core.pipeline_builder.VectorStoreCache"),
    ):
        embedder = MagicMock()
        embedder.model = "test-model"

        with pytest.raises(EmptyPDFError):
            await rag_system.build_pipeline("empty.pdf", "empty.pdf", embedder)


@pytest.mark.asyncio
async def test_aquery_success(rag_system):
    """정상적인 질의 응답 흐름 테스트"""
    mock_engine = AsyncMock()
    mock_engine.ainvoke.return_value = {
        "response": "답변입니다.",
        "thought": "생각 중...",
        "relevant_docs": [Document(page_content="근거", metadata={"page": 1})],
        "performance": {},
    }

    mock_monitor = MagicMock()
    mock_monitor.track_operation.return_value.__enter__ = MagicMock(return_value=None)
    mock_monitor.track_operation.return_value.__exit__ = MagicMock(return_value=False)
    mock_monitor.get_report.return_value = {}

    mock_op_type = MagicMock()
    mock_op_type.RAG_PIPELINE_TOTAL = "rag_total"

    with (
        patch("core.rag_core.SessionManager") as mock_session,
        patch(
            "core.rag_core.prepare_query_config_or_build", AsyncMock(return_value={})
        ),
        patch("core.rag_core.hydrate_documents") as mock_hydrate,
        patch("core.rag_core.get_resource_manager") as mock_manager,
        patch.object(
            rag_system, "_get_rag_engine", AsyncMock(return_value=mock_engine)
        ),
        patch(
            "services.monitoring.performance_monitor.get_performance_monitor",
            return_value=mock_monitor,
        ),
        patch("services.monitoring.performance_monitor.OperationType", mock_op_type),
        patch("core.graph_builder.format_context", return_value="context"),
    ):
        mock_session.get.return_value = mock_engine
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        result = await rag_system.aquery("질문")

        assert result["response"] == "답변입니다."
        mock_engine.ainvoke.assert_called_once()
        mock_hydrate.assert_called_once()


@pytest.mark.asyncio
async def test_aquery_not_ready(rag_system):
    """파이프라인 구축 전 질의 시 에러 발생 테스트"""
    with patch("core.rag_core.SessionManager") as mock_session:
        mock_session.get.return_value = None

        with pytest.raises(VectorStoreError):
            await rag_system.aquery("질문")
