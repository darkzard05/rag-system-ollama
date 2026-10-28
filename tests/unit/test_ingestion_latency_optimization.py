import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_load_pdf_docs_does_not_extract_words_eagerly():
    """RED Test: load_pdf_docs should not pass extract_words=True to to_markdown."""
    with (
        patch("pymupdf4llm.to_markdown") as mock_to_md,
        patch("src.core.document_processor.open_pdf_document") as mock_open,
        patch(
            "src.core.document_processor.compute_file_hash", return_value="dummy_hash"
        ),
    ):
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_open.return_value.__enter__.return_value = mock_doc
        mock_to_md.return_value = [{"text": "Sample content", "metadata": {"page": 1}}]

        from src.core.document_processor import load_pdf_docs

        await load_pdf_docs("dummy.pdf", "dummy.pdf")

        # extract_words must be False during initial ingestion
        assert mock_to_md.call_args[1].get("extract_words") is False


@pytest.mark.asyncio
async def test_semantic_chunker_cache_manager_disables_disk_cache():
    """RED Test: SemanticChunker should not enable disk cache for ephemeral sentence embeddings."""
    from src.core.semantic_chunker import EmbeddingBasedSemanticChunker

    mock_embedder = MagicMock()
    mock_embedder.model_name = "test_model"

    chunker = EmbeddingBasedSemanticChunker(embedder=mock_embedder)
    assert chunker.cache_manager.enable_disk_cache is False


@pytest.mark.asyncio
async def test_pipeline_builder_reuses_provided_file_hash():
    """RED Test: PipelineBuilder.build must not recompute hash if already provided."""
    from src.core.pipeline_builder import PipelineBuilder

    builder = PipelineBuilder(session_id="test_session")

    with (
        patch("src.core.pipeline_builder.compute_file_hash") as mock_hash,
        patch("src.core.pipeline_builder.VectorStoreCache") as mock_v_cache,
        patch.object(builder, "_build_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_impl.return_value = ("Success", False)
        provided_hash = "already_computed_sha256_hash_value_1234567890"

        await builder.build(
            file_path="dummy.pdf",
            file_name="dummy.pdf",
            embedder=MagicMock(),
            file_hash=provided_hash,
        )

        # compute_file_hash should NOT be called since hash was provided
        mock_hash.assert_not_called()
