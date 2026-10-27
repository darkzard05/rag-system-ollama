from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from src.core.chunking import split_documents


class MockSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            text = doc.page_content
            mid = len(text) // 2
            if mid > 0:
                chunks.append(
                    Document(page_content=text[:mid], metadata=dict(doc.metadata))
                )
                chunks.append(
                    Document(page_content=text[mid:], metadata=dict(doc.metadata))
                )
            else:
                chunks.append(Document(page_content=text, metadata=dict(doc.metadata)))
        return chunks


@pytest.mark.asyncio
@patch("src.core.chunking._recursive_splitter_cls")
async def test_metadata_inheritance(mock_splitter_cls):
    mock_splitter_cls.return_value = MockSplitter
    # 1. 준비: 메타데이터가 포함된 문서 생성
    # 일부러 아주 긴 텍스트를 넣어 분할이 일어나게 함
    content = "This is a document. " * 5
    metadata = {"file_path": "test/path/doc.pdf", "file_hash": "abc123hash", "page": 1}
    docs = [Document(page_content=content, metadata=metadata)]

    # 2. 실행: 문서 분할
    # embedder=None으로 설정하여 기본 텍스트 분할기 작동 확인
    chunks, _ = await split_documents(docs, embedder=None)

    # 3. 검증: 모든 청크가 핵심 메타데이터를 상속받았는지 확인
    assert len(chunks) > 0, "Chunks should be generated"
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.get("file_path") == "test/path/doc.pdf", (
            f"Chunk {i} missing file_path"
        )
        assert chunk.metadata.get("file_hash") == "abc123hash", (
            f"Chunk {i} missing file_hash"
        )


@pytest.mark.asyncio
@patch("src.core.chunking._recursive_splitter_cls")
async def test_metadata_inheritance_with_multiple_docs(mock_splitter_cls):
    mock_splitter_cls.return_value = MockSplitter
    # 여러 문서가 섞여 있을 때 각각의 메타데이터가 잘 유지되는지 확인
    docs = [
        Document(
            page_content="Doc 1 content ",
            metadata={"file_path": "doc1.pdf", "file_hash": "hash1"},
        ),
        Document(
            page_content="Doc 2 content ",
            metadata={"file_path": "doc2.pdf", "file_hash": "hash2"},
        ),
    ]

    chunks, _ = await split_documents(docs, embedder=None)

    # Doc 1에서 나온 청크들은 doc1.pdf를, Doc 2는 doc2.pdf를 가져야 함
    for chunk in chunks:
        path = chunk.metadata.get("file_path")
        hash_val = chunk.metadata.get("file_hash")
        if "Doc 1" in chunk.page_content:
            assert path == "doc1.pdf"
            assert hash_val == "hash1"
        elif "Doc 2" in chunk.page_content:
            assert path == "doc2.pdf"
            assert hash_val == "hash2"
