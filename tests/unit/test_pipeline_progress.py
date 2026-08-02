"""
RAG 파이프라인 구축 진행률(on_progress) 콜백 전달 검증 테스트.

빌드 파이프라인이 텍스트 추출 → 청킹 → 벡터/BM25 인덱스 → 최종화 단계를
0~100의 중간 진행률로 보고하는지 검증합니다.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.document_processor import _extraction_progress_pct, load_pdf_docs
from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_pipeline_builder_reports_intermediate_progress():
    """build()가 추출→청킹→인덱스→최종화 단계를 단조 증가하는 진행률로 보고하는지 검증합니다."""
    progress: list[int] = []

    def on_progress(pct: int) -> None:
        progress.append(pct)

    async def fake_load_pdf_docs(
        file_path: str, file_name: str, on_progress=None, **kwargs
    ):
        # 페이지 단위 추출 진행률을 시뮬레이션 (load_pdf_docs 내부 호출 대체)
        if on_progress:
            on_progress(5)
            on_progress(25)
        return [
            Document(page_content="first page content"),
            Document(page_content="second page content"),
        ]

    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)

    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash123"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch("core.pipeline_builder.VectorStoreCache.save"),
        patch("core.pipeline_builder.load_pdf_docs", new=fake_load_pdf_docs),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=([Document(page_content="chunk")], [])),
        ),
        patch("core.pipeline_builder.create_vector_store", return_value=MagicMock()),
        patch("core.pipeline_builder.create_bm25_retriever", return_value=MagicMock()),
        patch(
            "core.pipeline_builder.get_resource_manager",
            return_value=resource_manager,
        ),
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        builder = PipelineBuilder(session_id="test-progress")
        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
            on_progress=on_progress,
        )

    # 텍스트 추출 단계 (5~45): 첫 페이지 진행률이 전달되어야 함
    assert progress[0] == 5
    # 페이지 진행률이 파이프라인까지 전달되어야 함
    assert 25 in progress
    # 청킹(60), 인덱스(85), 최종화(100) 단계가 보고되어야 함
    assert 60 in progress
    assert 85 in progress
    assert 100 in progress
    # 전체 진행률은 단조 비감소해야 함
    assert progress == sorted(progress)
    assert len(set(progress)) >= 3


def test_load_pdf_docs_calls_on_progress_with_pct():
    """load_pdf_docs가 정수형 진행률(5~100)을 단조 증가로 보고하는지 검증합니다."""
    progress: list[int] = []

    def on_progress(pct: int) -> None:
        progress.append(pct)

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 10

    mock_chunks = [
        {"text": f"Page {i + 1} content", "metadata": {"page": i + 1}, "words": []}
        for i in range(10)
    ]

    with (
        patch("pymupdf4llm.to_markdown", return_value=mock_chunks),
        patch("core.document_processor.open_pdf_document") as mock_cm,
        patch("core.document_processor.compute_file_hash", return_value="hash123"),
        patch("core.document_processor.SessionManager"),
    ):
        mock_cm.return_value.__enter__.return_value = mock_doc
        mock_cm.return_value.__exit__.return_value = False

        docs = asyncio.run(
            load_pdf_docs(
                "fake.pdf",
                "fake.pdf",
                on_progress=on_progress,
                session_id="test-progress",
            )
        )

    assert len(docs) == 10
    assert progress, "on_progress는 한 번 이상 호출되어야 합니다"
    assert all(isinstance(p, int) and 5 <= p <= 100 for p in progress)
    assert progress == sorted(progress)


def test_load_pdf_docs_fallback_reports_per_page_progress():
    """C-Engine 폴백 경로에서 페이지별 진행률(5→45)이 단조 증가로 보고되는지 검증합니다."""
    progress: list[int] = []

    def on_progress(pct: int) -> None:
        progress.append(pct)

    page_mock = MagicMock()
    page_mock.get_text.side_effect = lambda mode: (
        "Fallback page content" if mode == "text" else []
    )

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 10
    mock_doc.__getitem__.return_value = page_mock

    with (
        patch("pymupdf4llm.to_markdown", side_effect=RuntimeError("layout failed")),
        patch("core.document_processor.open_pdf_document") as mock_cm,
        patch("core.document_processor.compute_file_hash", return_value="hash123"),
        patch("core.document_processor.SessionManager"),
    ):
        mock_cm.return_value.__enter__.return_value = mock_doc
        mock_cm.return_value.__exit__.return_value = False

        docs = asyncio.run(
            load_pdf_docs(
                "fake.pdf",
                "fake.pdf",
                on_progress=on_progress,
                session_id="test-progress",
            )
        )

    assert len(docs) == 10
    assert progress == sorted(progress)
    # 페이지 1개 처리 후 첫 진행률 (5% ~ 45% 범위 시작점)
    assert progress[0] == _extraction_progress_pct(1, 10)
    # 추출 완료 시 45%에 도달해야 함
    assert progress[-1] == 45
    assert all(5 <= p <= 45 for p in progress)
