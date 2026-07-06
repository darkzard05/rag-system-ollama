"""
Verification tests for specifically identified defects (P0-P2).
These tests ensure that critical fixes are not regressed.
"""

import pytest
import asyncio
import os
import psutil
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.rag_core import RAGSystem
from src.core.session import SessionManager
from src.core.model_loader import ModelManager
from tests.utils import mock_factory
from langchain_core.documents import Document


def get_open_files():
    """현재 프로세스가 열고 있는 파일 핸들 수를 반환합니다."""
    proc = psutil.Process()
    return len(proc.open_files())


@pytest.mark.asyncio
async def test_p0_1_pdf_handle_leak(session_context, test_pdf_path):
    """P0-1: PDF 파일 핸들 누수 검증"""
    rag_system = RAGSystem(session_id=session_context)

    pdf_path = test_pdf_path

    initial_files = get_open_files()

    # 여러 번 반복하여 누수 확인
    for _ in range(5):
        with (
            patch(
                "core.rag_core.split_documents",
                AsyncMock(return_value=([mock_factory.create_mock_document()], [])),
            ),
            patch(
                "core.rag_core.create_vector_store",
                return_value=mock_factory.create_mock_vector_store(),
            ),
            patch(
                "core.rag_core.create_bm25_retriever",
                return_value=mock_factory.create_mock_bm25_retriever(),
            ),
        ):
            embedder = mock_factory.create_mock_embedder()
            await rag_system.build_pipeline(pdf_path, "test.pdf", embedder)

    final_files = get_open_files()
    # 약간의 오차는 허용하되, 반복 횟수만큼 늘어나지 않아야 함
    assert final_files - initial_files <= 2, (
        f"PDF handle leak detected: {initial_files} -> {final_files}"
    )


@pytest.mark.asyncio
async def test_p0_2_session_sync_concurrency(session_context):
    """P0-2: 세션 동기화 및 스레드 안전성 검증"""
    rag_system = RAGSystem(session_id=session_context)

    # 동시에 많은 수의 set/get 요청을 보내어 Race Condition 확인
    async def worker(i):
        SessionManager.set(f"key_{i}", f"val_{i}", session_id=session_context)
        return SessionManager.get(f"key_{i}", session_id=session_context)

    results = await asyncio.gather(*(worker(i) for i in range(100)))

    for i, res in enumerate(results):
        assert res == f"val_{i}"


@pytest.mark.asyncio
async def test_p1_1_toctou_atomic_build(session_context):
    """P1-1: TOCTOU(Time-of-Check to Time-of-Use) 방지 및 원자적 빌드 검증"""
    # ModelManager.get_or_build_resource가 중복 빌드를 방지하는지 확인
    file_hash = "atomic_test_hash"
    build_count = 0

    async def mock_build(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.1)  # 빌드 시간 시뮬레이션
        return (
            mock_factory.create_mock_vector_store(),
            mock_factory.create_mock_bm25_retriever(),
        )

    # 여러 태스크가 동시에 동일 리소스를 요청
    tasks = [
        ModelManager.get_or_build_resource(
            file_hash,
            build_fn=mock_build,
            file_path="dummy.pdf",
            file_name="dummy.pdf",
            embedder=mock_factory.create_mock_embedder(),
        )
        for _ in range(10)
    ]

    await asyncio.gather(*tasks)

    # 빌드 함수는 단 한 번만 호출되어야 함
    assert build_count == 1, (
        f"Resource build was not atomic: called {build_count} times"
    )


@pytest.mark.asyncio
async def test_p2_2_metadata_classification(session_context):
    """P2-2: 메타데이터 참고문헌 분류 정확성 검증"""
    # 이 테스트는 실제 core.graph_builder.format_context나
    # RAGSystem.aquery 결과의 documents 메타데이터를 검증해야 함
    rag_system = RAGSystem(session_id=session_context)

    # 모킹된 엔진 응답 설정
    mock_engine = AsyncMock()
    mock_engine.ainvoke.return_value = {
        "response": "답변",
        "relevant_docs": [
            Document(page_content="내용1", metadata={"page": 1, "source": "doc1"}),
            Document(page_content="내용2", metadata={"page": 2, "source": "doc1"}),
        ],
    }

    with (
        patch("core.rag_core.SessionManager") as mock_session,
        patch.object(
            rag_system, "_prepare_config", AsyncMock(return_value={"configurable": {}})
        ),
        patch.object(
            rag_system, "_get_rag_engine", AsyncMock(return_value=mock_engine)
        ),
    ):
        mock_session.get.side_effect = (
            lambda k, **kwargs: mock_engine if k == "rag_engine" else None
        )

        result = await rag_system.aquery("질문")

        # 결과에 documents가 포함되어 있고, 메타데이터가 유지되는지 확인
        assert len(result["documents"]) == 2
        assert result["documents"][0].metadata["page"] == 1
        assert result["documents"][1].metadata["page"] == 2


from langchain_core.documents import Document
from unittest.mock import patch
