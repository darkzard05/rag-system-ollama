"""P1: PipelineBuilder 타입 이벤트 발행 계약 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.build_state import BuildEvent, BuildPhase
from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_build_emits_typed_events_with_phases() -> None:
    events: list[BuildEvent] = []
    legacy: list[int] = []

    async def fake_load_pdf_docs(
        file_path: str, file_name: str, on_progress=None, **kwargs
    ):
        if on_progress is not None:
            try:
                on_progress(15, detail="Extracting...")
            except TypeError:
                on_progress(15)
        return [Document(page_content="content")]

    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)
    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash_ev1"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch("core.pipeline_builder.VectorStoreCache.save"),
        patch("core.pipeline_builder.load_pdf_docs", new=fake_load_pdf_docs),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=([Document(page_content="c")], [])),
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
        builder = PipelineBuilder(session_id="test-events-phases")
        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
            on_progress=legacy.append,
            on_event=events.append,
        )

    assert events, "on_event 콜백이 호출되지 않았다"
    phases = {e.phase for e in events}
    assert BuildPhase.EXTRACTING in phases
    assert BuildPhase.CHUNKING in phases
    assert BuildPhase.INDEXING in phases
    assert BuildPhase.REGISTERING in phases
    for e in events:
        assert e.detail_key, "detail_key가 비어 있다"
        assert e.status, "레거시 status 표시문이 비어 있다"
    # 레거시 콜백도 함께 발행된다 (이중 발행 계약)
    assert legacy, "레거시 on_progress가 호출되지 않았다"


@pytest.mark.asyncio
async def test_build_cancel_event_aborts() -> None:
    cancel = asyncio.Event()
    cancel.set()
    embedder = MagicMock()
    embedder.model = "fake-model"
    embedder.model_name = "fake-model"
    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash_ev2"),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
        patch(
            "core.pipeline_builder.load_pdf_docs",
            new=AsyncMock(return_value=[Document(page_content="c")]),
        ),
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=([Document(page_content="c")], [])),
        ),
    ):
        builder = PipelineBuilder(session_id="test-events-cancel")
        with pytest.raises(asyncio.CancelledError):
            await builder.build(
                file_path="fake.pdf",
                file_name="fake.pdf",
                embedder=embedder,
                cancel=cancel,
            )
