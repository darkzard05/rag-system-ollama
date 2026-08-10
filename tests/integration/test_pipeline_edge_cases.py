import asyncio
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from common.exceptions import EmptyPDFError, PDFProcessingError
from core.rag_core import RAGSystem
from core.session import SessionManager


def create_mock_document(
    content: str = "테스트 문서 내용", metadata: dict = None
) -> Document:
    """표준 테스트 문서 객체를 생성합니다."""
    if metadata is None:
        metadata = {"page": 1, "file_hash": "test_hash", "has_coordinates": True}
    return Document(page_content=content, metadata=metadata)


def create_mock_embedder():
    """표준 모킹 임베더를 생성합니다."""
    mock_embedder = MagicMock()
    mock_embedder.model = "test-embedding-model"
    # embed_documents mock
    mock_embedder.embed_documents.return_value = [[0.1] * 128]
    # embed_query mock
    mock_embedder.embed_query.return_value = [0.1] * 128
    return mock_embedder


def create_mock_vector_store():
    """표준 모킹 벡터 스토어를 생성합니다."""
    mock_vs = MagicMock()

    # as_retriever mock
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    # similarity_search mock
    mock_vs.similarity_search.return_value = [create_mock_document()]

    return mock_vs


def create_mock_bm25_retriever():
    """표준 모킹 BM25 리트리버를 생성합니다."""
    mock_bm25 = MagicMock()
    mock_bm25.k = 5
    mock_bm25.get_relevant_documents.return_value = [create_mock_document()]
    return mock_bm25


class TestPipelineEdgeCases:
    """
    RAG 파이프라인의 엣지 케이스를 검증하는 통합 테스트 클래스.
    빈 파일, 손상된 파일, 타임아웃, 비정상 쿼리 등을 처리합니다.
    """

    @pytest.fixture
    def rag_system(self):
        """세션 상태(SessionManager)를 사용하는 RAGSystem 인스턴스를 제공합니다."""
        session_id = "edge_case_test_session"
        rag = RAGSystem(session_id=session_id)
        SessionManager.set("llm", MagicMock(), session_id=session_id)
        SessionManager.set("embedder", MagicMock(), session_id=session_id)
        return rag

    @pytest.fixture
    def mock_embedder(self):
        """표준 모킹 임베더를 제공합니다."""
        return create_mock_embedder()

    @pytest.mark.asyncio
    async def test_empty_pdf(self, rag_system: RAGSystem, mock_embedder: Any):
        """
        엣지 케이스 1: 텍스트가 전혀 없는 빈 PDF 파일 처리 검증.
        PDF 헤더는 존재하지만 실제 내용이 없는 경우 EmptyPDFError 또는 PDFProcessingError가 발생해야 합니다.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            _ = tmp.write(b"%PDF-1.4\n%EOF")
            tmp_path = tmp.name

        try:
            await rag_system.build_pipeline(tmp_path, "empty.pdf", mock_embedder)
            pytest.fail("Should have raised EmptyPDFError or PDFProcessingError")
        except (EmptyPDFError, PDFProcessingError):
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__} - {e}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_corrupted_pdf(self, rag_system: RAGSystem, mock_embedder: Any):
        """
        엣지 케이스 2: 손상된 PDF 파일 처리 검증.
        바이너리가 깨진 파일의 경우 EmptyPDFError 또는 PDFProcessingError가 발생해야 합니다.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            _ = tmp.write(os.urandom(1024))
            tmp_path = tmp.name

        try:
            await rag_system.build_pipeline(tmp_path, "corrupted.pdf", mock_embedder)
            pytest.fail("Should have raised EmptyPDFError or PDFProcessingError")
        except (EmptyPDFError, PDFProcessingError):
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__} - {e}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_non_existent_file(self, rag_system: RAGSystem, mock_embedder: Any):
        """
        엣지 케이스 3: 존재하지 않는 파일 경로 처리 검증.
        유효하지 않은 경로 전달 시 FileNotFoundError 또는 PDFProcessingError가 발생해야 합니다.
        """
        fake_path = "non_existent_file_12345.pdf"

        try:
            await rag_system.build_pipeline(fake_path, "fake.pdf", mock_embedder)
            pytest.fail("Should have raised FileNotFoundError or PDFProcessingError")
        except (FileNotFoundError, PDFProcessingError):
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__} - {e}")

    @pytest.mark.asyncio
    async def test_empty_query(self, rag_system: RAGSystem):
        """
        엣지 케이스 4: 빈 쿼리/질문 처리 검증.
        빈 문자열 전달 시 시스템이 크래시 없이 적절한 응답을 반환하거나 예외를 처리해야 합니다.
        """
        mock_vs = create_mock_vector_store()
        mock_bm25 = create_mock_bm25_retriever()

        mock_engine = AsyncMock()
        mock_engine.ainvoke.return_value = {
            "response": "빈 쿼리 응답",
            "thought": "생각",
            "relevant_docs": [],
            "performance": {},
        }

        with (
            patch("core.rag_core.get_resource_manager") as mock_manager,
            patch("core.pipeline_builder.get_resource_manager", mock_manager),
            patch.object(
                rag_system, "_get_rag_engine", AsyncMock(return_value=mock_engine)
            ),
        ):
            mock_manager.return_value.retrievers.get.return_value = (mock_vs, mock_bm25)
            # [T18-스모크] T12(R1b-04) async get_retrievers(pin 경유) 호환 mock —
            # bare MagicMock은 await 불가하므로 AsyncMock으로 동일 쌍을 반환한다.
            mock_manager.return_value.get_retrievers = AsyncMock(
                return_value=(mock_vs, mock_bm25)
            )
            SessionManager.set(
                "file_hash", "dummy_hash", session_id=rag_system.session_id
            )
            SessionManager.set(
                "pdf_file_path", "dummy.pdf", session_id=rag_system.session_id
            )

            result = await rag_system.aquery("")
            assert "response" in result

    @pytest.mark.asyncio
    async def test_model_timeout(self, rag_system: RAGSystem):
        """
        엣지 케이스 5: LLM 타임아웃 시나리오 검증.
        모델 응답 생성 중 타임아웃이 발생했을 때 시스템이 이를 적절히 처리하는지 확인합니다.
        """
        mock_vs = create_mock_vector_store()
        mock_bm25 = create_mock_bm25_retriever()

        # astream_events가 asyncio.TimeoutError를 발생시키는 비동기 제너레이터
        async def mock_astream_events_gen(*args, **kwargs):
            raise asyncio.TimeoutError("Model timeout")
            yield None

        mock_engine = MagicMock()
        # AsyncMock 대신 MagicMock을 사용하여 제너레이터 반환을 명확히 함
        mock_engine.astream_events.side_effect = (
            lambda *args, **kwargs: mock_astream_events_gen(*args, **kwargs)
        )

        with (
            patch("core.rag_core.get_resource_manager") as mock_manager,
            patch("core.pipeline_builder.get_resource_manager", mock_manager),
            patch.object(
                rag_system, "_get_rag_engine", AsyncMock(return_value=mock_engine)
            ),
        ):
            mock_manager.return_value.retrievers.get.return_value = (mock_vs, mock_bm25)
            # [T18-스모크] T12(R1b-04) async get_retrievers(pin 경유) 호환 mock —
            # bare MagicMock은 await 불가하므로 AsyncMock으로 동일 쌍을 반환한다.
            mock_manager.return_value.get_retrievers = AsyncMock(
                return_value=(mock_vs, mock_bm25)
            )
            SessionManager.set(
                "file_hash", "timeout_hash", session_id=rag_system.session_id
            )
            SessionManager.set(
                "pdf_file_path", "timeout.pdf", session_id=rag_system.session_id
            )

            # astream은 제너레이터를 반환하므로, 반복문을 통해 실행해야 에러가 발생함
            stream = await rag_system.astream("타임아웃 테스트 질문")

            # _producer에서 asyncio.TimeoutError는 catch되지 않으므로
            # 반복문 실행 시 StopAsyncIteration 또는 TimeoutError가 발생할 수 있음
            # 시스템 설계상 이 에러가 어떻게 전파되는지 확인
            with pytest.raises(asyncio.TimeoutError):
                async for _ in stream:
                    pass

    @pytest.mark.asyncio
    async def test_very_long_query(self, rag_system: RAGSystem):
        """
        엣지 케이스 6: 매우 긴 쿼리 처리 검증.
        10,000자 이상의 매우 긴 쿼리를 전달했을 때 시스템이 크래시 없이 처리하는지 확인합니다.
        """
        mock_vs = create_mock_vector_store()
        mock_bm25 = create_mock_bm25_retriever()

        long_query = "A" * 10001

        # aquery가 정상 작동하도록 엔진 모킹
        mock_engine = AsyncMock()
        mock_engine.ainvoke.return_value = {
            "response": "긴 쿼리에 대한 답변입니다.",
            "thought": "긴 쿼리를 분석했습니다.",
            "relevant_docs": [],
            "performance": {},
        }

        with (
            patch("core.rag_core.get_resource_manager") as mock_manager,
            patch("core.pipeline_builder.get_resource_manager", mock_manager),
            patch.object(
                rag_system, "_get_rag_engine", AsyncMock(return_value=mock_engine)
            ),
        ):
            mock_manager.return_value.retrievers.get.return_value = (mock_vs, mock_bm25)
            # [T18-스모크] T12(R1b-04) async get_retrievers(pin 경유) 호환 mock —
            # bare MagicMock은 await 불가하므로 AsyncMock으로 동일 쌍을 반환한다.
            mock_manager.return_value.get_retrievers = AsyncMock(
                return_value=(mock_vs, mock_bm25)
            )
            SessionManager.set(
                "file_hash", "long_query_hash", session_id=rag_system.session_id
            )
            SessionManager.set(
                "pdf_file_path", "long_query.pdf", session_id=rag_system.session_id
            )

            # 쿼리가 매우 길어도 에러 없이 응답을 반환해야 함
            result = await rag_system.aquery(long_query)
            assert "response" in result
            assert result["response"] == "긴 쿼리에 대한 답변입니다."
