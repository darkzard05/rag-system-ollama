from langchain_core.documents import Document
from src.core.chunking import _postprocess_metadata


def test_postprocess_metadata_audit():
    split_docs = [
        Document(page_content="Chunk 1", metadata={"start_index": 0}),
        Document(page_content="Chunk 2", metadata={"start_index": 20}),
    ]

    _postprocess_metadata(split_docs)

    for i, chunk in enumerate(split_docs):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["is_content"] is True
        assert chunk.metadata["is_reference"] is False


def test_postprocess_metadata_multi_doc_audit():
    split_docs = [
        Document(page_content="Normal content"),
        Document(page_content="## References\n[1] Smith et al."),
        Document(page_content="Another reference"),
    ]

    _postprocess_metadata(split_docs)

    assert split_docs[0].metadata["is_reference"] is False
    assert split_docs[0].metadata["is_content"] is True
    assert split_docs[1].metadata["is_reference"] is True
    assert split_docs[2].metadata["is_reference"] is True
