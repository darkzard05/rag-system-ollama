import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.pipeline_builder import PipelineBuilder


@pytest.mark.asyncio
async def test_build_reuses_provided_file_hash():
    builder = PipelineBuilder(session_id="test")

    with (
        patch("core.pipeline_builder.compute_file_hash", return_value="hash123") as mock_hash,
        patch("core.pipeline_builder.load_pdf_docs", new=AsyncMock(return_value=["doc"])) as mock_load,
        patch("core.pipeline_builder.split_documents", new=AsyncMock(return_value=(["chunk"], None))) as mock_split,
        patch("core.pipeline_builder.create_vector_store", return_value="vs"),
        patch("core.pipeline_builder.create_bm25_retriever", return_value="bm25"),
        patch("core.pipeline_builder.get_resource_manager") as mock_resource_manager,
        patch("core.pipeline_builder.build_graph", new=AsyncMock(return_value="workflow")),
        patch("core.pipeline_builder.VectorStoreCache.load", return_value=(None, None, None)),
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
