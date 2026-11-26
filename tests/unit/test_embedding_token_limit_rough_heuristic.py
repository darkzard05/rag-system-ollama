"""The rough token estimate must not skip the real embedding token limit."""

import pytest
from langchain_core.documents import Document

from core import chunking


class _FakeSplitter:
    def __init__(self, **kwargs: object) -> None:  # noqa: ARG002
        return None

    def split_documents(self, docs: list[Document]) -> list[Document]:  # noqa: ARG002
        return [
            Document(page_content="!" * 200, metadata={"page": 1}),
            Document(page_content="!" * 200, metadata={"page": 1}),
        ]


@pytest.mark.asyncio
async def test_punctuation_heavy_chunk_is_checked_and_split(monkeypatch):
    """A chunk under rough*1.6 can still exceed the real tokenizer limit."""
    doc = Document(page_content="!" * 700, metadata={"page": 1})

    monkeypatch.setattr(
        chunking, "TEXT_SPLITTER_CONFIG", {"embedding_token_limit": 448}
    )
    monkeypatch.setattr(chunking, "_embedding_token_counts", lambda texts: [600])
    monkeypatch.setattr(
        chunking, "_embedding_token_count", lambda text: len(text) // 3 + 1
    )
    monkeypatch.setattr(chunking, "_recursive_splitter_cls", lambda: _FakeSplitter)

    out_docs, _ = await chunking._enforce_embedding_token_limit([doc], None, None)

    assert len(out_docs) == 2
    assert all(d.metadata["split_from_oversized"] for d in out_docs)
