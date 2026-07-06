import pytest
from langchain_core.documents import Document
from src.core.chunking import _postprocess_metadata


def test_postprocess_metadata_audit():
    # 1. 준비: 원본 문서 (핵심 메타데이터 보유)
    original_docs = [
        Document(
            page_content="First page content",
            metadata={"file_path": "doc1.pdf", "file_hash": "hash1"},
        ),
        Document(
            page_content="Second page content",
            metadata={"file_path": "doc1.pdf", "file_hash": "hash1"},
        ),
    ]

    # 2. 시뮬레이션: 분할된 청크들이 메타데이터를 유실한 상태
    # start_index는 유지되어야 매핑 가능
    split_docs = [
        Document(
            page_content="Chunk 1", metadata={"start_index": 0}
        ),  # file_path, file_hash 유실
        Document(
            page_content="Chunk 2", metadata={"start_index": 20}
        ),  # file_path, file_hash 유실
    ]

    # 3. 실행: 메타데이터 후처리 (감사 로직 작동)
    _postprocess_metadata(split_docs, original_docs)

    # 4. 검증: 유실되었던 메타데이터가 복구되었는지 확인
    for chunk in split_docs:
        assert chunk.metadata.get("file_path") == "doc1.pdf", (
            "file_path should be recovered"
        )
        assert chunk.metadata.get("file_hash") == "hash1", (
            "file_hash should be recovered"
        )


def test_postprocess_metadata_multi_doc_audit():
    # 여러 문서가 섞여 있을 때 정확한 문서에서 복구하는지 확인
    original_docs = [
        Document(
            page_content="Doc A content",
            metadata={"file_path": "A.pdf", "file_hash": "hashA"},
        ),
        Document(
            page_content="Doc B content",
            metadata={"file_path": "B.pdf", "file_hash": "hashB"},
        ),
    ]

    # Doc A의 오프셋(0)과 Doc B의 오프셋(15+)을 가지는 청크들
    split_docs = [
        Document(page_content="Chunk A", metadata={"start_index": 0}),
        Document(page_content="Chunk B", metadata={"start_index": 20}),
    ]

    _postprocess_metadata(split_docs, original_docs)

    assert split_docs[0].metadata.get("file_path") == "A.pdf"
    assert split_docs[1].metadata.get("file_path") == "B.pdf"
