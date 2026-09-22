"""DEFECT #2: 콜드 리인덱스 진행률이 너무 성긴 단계(0→60→85→100)로 보고되는 테스트.

현재 구현: PipelineBuilder._build_impl이 on_progress를 4번만 호출한다:
- 0 (시작)
- 60 (청킹 완료)
- 85 (벡터/BM25 인덱스 완료)
- 100 (최종화 완료)

사용자는 "문서 분석 중입니다..." 스피너만 보며, 몇 분 동안 진행 상황을 모른다.

목표 상태: 최소 8개 이상의 세분화된 단계(파싱→청크→임베딩 배치→FAISS→BM25→리랭커→LLM 프리로드→최종)
+ 경과 시간 + 현재 단계 라벨이 표시되어야 한다.

이 테스트는 현재 코드에서 반드시 실패해야 한다 (TDD 레드).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_build_progress_has_at_least_8_stages():
    """build()가 최소 8개의 세분화된 진행률 단계를 보고해야 한다.

    현재 실패 원인: on_progress가 4번만 호출된다 (0, 60, 85, 100).
    목표: 파싱→청크→임베딩 배치→FAISS→BM25→리랭커→LLM 프리로드→최종
    최소 8개 단계.
    """
    progress_calls: list[int] = []

    def on_progress(pct: int) -> None:
        progress_calls.append(pct)

    async def fake_load_pdf_docs(
        file_path: str, file_name: str, on_progress=None, **kwargs
    ):
        if on_progress:
            on_progress(5)  # 파싱 시작
            on_progress(15)  # 파싱 중
            on_progress(25)  # 파싱 완료
        return [
            Document(page_content="page 1"),
            Document(page_content="page 2"),
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
        builder = PipelineBuilder(session_id="test-progress-granular")
        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
            on_progress=on_progress,
        )

    # 현재: 4~5번 호출 (0, 5, 15, 25, 60, 85, 100 = ~7번)
    # 목표: 최소 8개 고유 단계
    unique_progress = sorted(set(progress_calls))
    assert len(unique_progress) >= 8, (
        f"진행률 단계가 {len(unique_progress)}개뿐이다 (목표: 최소 8개). "
        f"현재 단계: {unique_progress}"
    )


@pytest.mark.asyncio
async def test_build_progress_includes_stage_labels():
    """각 진행률 단계에 라벨(텍스트 설명)이 포함되어야 한다.

    현재 실패 원인: on_progress(pct)는 정수만 전달하고, 어떤 단계인지
    알리는 텍스트 정보가 없다.

    목표: ProgressEvent(stage, pct, detail, elapsed) 같은 구조체가 필요하다.
    """
    progress_details: list[dict] = []

    def on_progress(pct: int, detail: str = "") -> None:
        progress_details.append({"pct": pct, "detail": detail})

    async def fake_load_pdf_docs(
        file_path: str, file_name: str, on_progress=None, **kwargs
    ):
        if on_progress:
            on_progress(5, "Extracting text from PDF...")
        return [Document(page_content="content")]

    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)

    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash456"),
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
        builder = PipelineBuilder(session_id="test-progress-labels")
        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
            on_progress=on_progress,
        )

    # on_progress에 detail 문자열이 전달되어야 한다
    # 현재: on_progress(pct: int) 시그니처이므로 detail이 전달되지 않음
    has_details = any(d["detail"] for d in progress_details)
    assert has_details, (
        "진행률 단계에 라벨(detail)이 포함되지 않았다. "
        f"현재 콜백: {progress_details[:3]}"
    )


@pytest.mark.asyncio
async def test_build_progress_is_monotonically_increasing():
    """진행률이 단조 증가해야 한다 (감소하지 않음)."""
    progress_calls: list[int] = []

    def on_progress(pct: int) -> None:
        progress_calls.append(pct)

    async def fake_load_pdf_docs(
        file_path: str, file_name: str, on_progress=None, **kwargs
    ):
        if on_progress:
            on_progress(10)
            on_progress(30)
        return [Document(page_content="content")]

    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)

    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash789"),
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
        builder = PipelineBuilder(session_id="test-progress-mono")
        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
            on_progress=on_progress,
        )

    assert progress_calls == sorted(progress_calls), (
        f"진행률이 감소했다: {progress_calls}"
    )


@pytest.mark.asyncio
async def test_build_cancel_rolls_back_flags():
    """빌드 취소 시 is_building_rag가 False로 리셋되어야 한다.

    현재 실패 원인: 취소 시 플래그 롤백이 제대로 동작하지 않을 수 있다.
    """
    from core.session import SessionManager

    SessionManager.set("is_building_rag", True, session_id="test-cancel")
    SessionManager.set("needs_rag_rebuild", True, session_id="test-cancel")

    async def fake_load_pdf_docs(*args, **kwargs):
        raise asyncio.CancelledError("Build cancelled")

    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)

    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash_cancel"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch("core.pipeline_builder.load_pdf_docs", new=fake_load_pdf_docs),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=([], [])),
        ),
        patch(
            "core.pipeline_builder.get_resource_manager",
            return_value=resource_manager,
        ),
    ):
        builder = PipelineBuilder(session_id="test-cancel")
        with pytest.raises(asyncio.CancelledError):
            await builder.build(
                file_path="fake.pdf",
                file_name="fake.pdf",
                embedder=embedder,
            )

    # 취소 후 플래그가 리셋되어야 한다
    assert (
        SessionManager.get("is_building_rag", False, session_id="test-cancel") is False
    )
