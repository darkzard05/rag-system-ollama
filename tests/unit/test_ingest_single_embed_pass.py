"""Regression test: split_documents must invoke the embedder exactly ONCE.

T15 micro-win: confirms a single embed pass per chunk set (no redundant
per-chunk embed), guarding the single-embedder-reuse acceptance criterion.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.core.chunking import split_documents


def _make_doc(
    content: str = "테스트 문서 내용", metadata: dict | None = None
) -> Document:
    if metadata is None:
        metadata = {"page": 1, "file_hash": "test_hash", "has_coordinates": True}
    return Document(page_content=content, metadata=metadata)


def _make_embedder() -> MagicMock:
    """Embedder whose embed_documents returns one vector per input chunk."""
    emb = MagicMock()
    emb.model = "test-embedding-model"

    def _embed(texts):
        # One 128-dim vector per input chunk — mirrors real embedder contract.
        return [[0.1] * 128 for _ in texts]

    emb.embed_documents.side_effect = _embed
    emb.embed_query.return_value = [0.1] * 128
    return emb


@pytest.mark.asyncio
async def test_split_documents_single_embed_pass():
    """A multi-chunk doc set must be embedded exactly once via embed_documents.

    This guards the T15 acceptance: split_documents reuses the single embedder
    instance and runs one embedding pass (no second per-chunk embed call).
    """
    # 30 chunks to also exercise a realistic large-PDF chunk count.
    docs = [_make_doc(content=f"섹션 {i} 내용") for i in range(30)]
    embedder = _make_embedder()

    with (
        patch("src.core.chunking.SEMANTIC_CHUNKER_CONFIG", {"enabled": False}),
        patch(
            "src.core.chunking.TEXT_SPLITTER_CONFIG",
            {"chunk_size": 500, "chunk_overlap": 50},
        ),
        patch("src.core.chunking.RecursiveCharacterTextSplitter") as mock_splitter_cls,
        patch("src.core.chunking.SessionManager.add_status_log"),
    ):
        mock_splitter_inst = mock_splitter_cls.return_value
        sql_split = mock_splitter_inst.split_documents
        sql_split.return_value = [_make_doc(content=f"조각 {i}") for i in range(30)]

        split_docs, vectors = await split_documents(docs, embedder=embedder)

    assert len(split_docs) == 30
    assert vectors is not None
    assert len(vectors) == 30
    # The single embed pass: embed_documents must be called exactly ONCE,
    # not once per chunk.
    assert embedder.embed_documents.call_count == 1
    np.testing.assert_array_equal(
        embedder.embed_documents.call_args.args[0],
        [f"조각 {i}" for i in range(30)],
    )
