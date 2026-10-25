from langchain_core.documents import Document

from core.graph._retrieval import _merge_adjacent_chunks


def test_chunk_index_none_regression():
    docs = [
        Document(
            page_content="Test content 1",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "chunk_index": None,
                "rerank_score": None,
            },
        ),
        Document(
            page_content="Test content 2",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 1,
                "rerank_score": 0.9,
            },
        ),
    ]

    # Should not raise TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
    merged = _merge_adjacent_chunks(docs, max_tokens=1200)
    assert isinstance(merged, list)
    assert len(merged) > 0
