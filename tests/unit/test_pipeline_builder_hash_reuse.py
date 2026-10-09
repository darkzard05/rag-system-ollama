from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_build_reuses_provided_file_hash():
    builder = PipelineBuilder(session_id="test")

    with (
        patch(
            "core.pipeline_builder.compute_file_hash", return_value="hash123"
        ) as mock_hash,
        patch(
            "core.pipeline_builder.load_pdf_docs", new=AsyncMock(return_value=["doc"])
        ) as mock_load,
        patch(
            "core.pipeline_builder.split_documents",
            new=AsyncMock(return_value=(["chunk"], None)),
        ),
        patch("core.pipeline_builder.create_vector_store", return_value="vs"),
        patch("core.pipeline_builder.create_bm25_retriever", return_value="bm25"),
        patch("core.pipeline_builder.get_resource_manager") as mock_resource_manager,
        patch(
            "core.pipeline_builder.build_graph", new=AsyncMock(return_value="workflow")
        ),
        patch(
            "core.pipeline_builder.VectorStoreCache.load",
            return_value=(None, None, None),
        ),
    ):
        resource_manager = MagicMock()
        resource_manager.register_retrievers = AsyncMock(return_value=None)
        mock_resource_manager.return_value = resource_manager

        embedder = MagicMock()
        embedder.model = "fake-model"
        embedder.model_name = "fake-model"

        await builder.build(
            file_path="fake.pdf",
            file_name="fake.pdf",
            embedder=embedder,
        )

        mock_hash.assert_called_once_with("fake.pdf")
        assert mock_load.await_count == 1
        _, kwargs = mock_load.await_args
        assert kwargs["file_hash"] == "hash123"


@pytest.mark.asyncio
async def test_register_and_finalize_invalidates_graph_on_fresh_build():
    """신규 인덱싱(fresh_build=True)은 build_graph 전에 그래프 캐시 무효화."""
    builder = PipelineBuilder(session_id="test-fresh")
    with (
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value="workflow"),
        ),
        patch("core.pipeline_builder.invalidate_graph_cache") as mock_invalidate,
        patch("core.pipeline_builder.get_resource_manager") as mock_rm,
        patch("cache.vector_cache.EngineCacheManager.set_engine"),
    ):
        rm = MagicMock()
        rm.register_retrievers = AsyncMock(return_value=None)
        mock_rm.return_value = rm
        await builder._register_and_finalize("hash1", "vs", "bm25", fresh_build=True)
        mock_invalidate.assert_called_once_with()


@pytest.mark.asyncio
async def test_register_and_finalize_keeps_graph_cache_on_reuse():
    """캐시 재사용(fresh_build=False)은 그래프 캐시 히트를 유지."""
    builder = PipelineBuilder(session_id="test-reuse")
    with (
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value="workflow"),
        ),
        patch("core.pipeline_builder.invalidate_graph_cache") as mock_invalidate,
        patch("core.pipeline_builder.get_resource_manager") as mock_rm,
        patch("cache.vector_cache.EngineCacheManager.set_engine"),
    ):
        rm = MagicMock()
        rm.register_retrievers = AsyncMock(return_value=None)
        mock_rm.return_value = rm
        await builder._register_and_finalize("hash1", "vs", "bm25", fresh_build=False)
        mock_invalidate.assert_not_called()
