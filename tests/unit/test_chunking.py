import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.documents import Document
from src.core.chunking import split_documents, _postprocess_metadata
from tests.utils.mock_factory import create_mock_document, create_mock_embedder

@pytest.mark.asyncio
async def test_split_documents_empty_list():
    """Empty document list should return empty list and None for vectors."""
    docs, vectors = await split_documents([])
    assert docs == []
    assert vectors is None

@pytest.mark.asyncio
async def test_split_documents_already_chunked_no_overflow():
    """Documents already chunked without overflow should be returned as is."""
    doc = create_mock_document(content="chunk 1", metadata={"is_already_chunked": True})
    docs = [doc]
    
    # Mock embedding cache to return a vector
    mock_vector = np.array([0.1, 0.2, 0.3])
    
    with patch("src.core.chunking.embedding_cache") as mock_cache, \
         patch("src.core.chunking.SessionManager.add_status_log") as mock_log:

        
        mock_cache.get_embeddings.return_value = ([mock_vector], [])
        
        split_docs, vectors = await split_documents(docs, embedder=create_mock_embedder())
        
        assert len(split_docs) == 1
        assert split_docs[0].page_content == "chunk 1"
        assert vectors is not None
        np.testing.assert_array_equal(vectors[0], mock_vector)
        mock_log.assert_any_call("기존 분할 구조 활용 (1개 섹션)", session_id=None)



@pytest.mark.asyncio
async def test_split_documents_already_chunked_with_overflow():
    """Documents already chunked but with overflow should trigger sub-chunking."""
    # max_chunk_size is 500. 500 * 1.5 = 750.
    long_content = "a" * 1000
    doc = create_mock_document(content=long_content, metadata={"is_already_chunked": True})
    docs = [doc]
    
    with patch("src.core.chunking.SEMANTIC_CHUNKER_CONFIG", {"enabled": False}), \
         patch("src.core.chunking.TEXT_SPLITTER_CONFIG", {"chunk_size": 500, "chunk_overlap": 50}), \
         patch("src.core.chunking.RecursiveCharacterTextSplitter") as mock_splitter_cls, \
         patch("src.core.chunking.SessionManager.add_status_log") as mock_log:

        
        mock_splitter_inst = mock_splitter_cls.return_value
        mock_splitter_inst.split_documents.return_value = [
            create_mock_document(content="sub-chunk 1"), 
            create_mock_document(content="sub-chunk 2")
        ]
        
        split_docs, vectors = await split_documents(docs, embedder=create_mock_embedder())
        
        assert len(split_docs) >= 2
        assert split_docs[0].page_content == "sub-chunk 1"
        mock_log.assert_any_call("대형 섹션 감지: 정밀 검색을 위한 하위 분할 시작", session_id=None)


@pytest.mark.asyncio
async def test_split_documents_standard_recursive():
    """Standard recursive splitting when not already chunked."""
    doc = create_mock_document(content="Standard text content")
    docs = [doc]
    
    mock_embedder = create_mock_embedder()
    # Mock embedder.embed_documents to return a list of vectors
    mock_embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    
    with patch("src.core.chunking.TEXT_SPLITTER_CONFIG", {"chunk_size": 500, "chunk_overlap": 50}), \
         patch("src.core.chunking.RecursiveCharacterTextSplitter.split_documents", return_value=[doc]), \
         patch("src.core.chunking.SessionManager.add_status_log") as mock_log:

        
        split_docs, vectors = await split_documents(docs, embedder=mock_embedder)
        
        assert len(split_docs) == 1
        assert vectors is not None
        assert isinstance(vectors[0], np.ndarray)
        mock_log.assert_any_call("문서 분할 및 문맥 추출 중...", session_id=None)


@pytest.mark.asyncio
async def test_split_documents_semantic_chunking():
    """Semantic chunking flow when enabled."""
    doc = create_mock_document(content="Semantic text content")
    docs = [doc]
    
    mock_embedder = create_mock_embedder()
    
    with patch("src.core.chunking.SEMANTIC_CHUNKER_CONFIG", {"enabled": True}), \
         patch("src.core.chunking.EmbeddingBasedSemanticChunker") as mock_chunker_class, \
         patch("src.core.chunking.SessionManager.add_status_log") as mock_log:

        
        mock_chunker_instance = mock_chunker_class.return_value
        mock_chunker_instance.split_documents = AsyncMock(return_value=([doc], [np.array([0.1])]))
        
        split_docs, vectors = await split_documents(docs, embedder=mock_embedder)
        
        assert len(split_docs) == 1
        assert vectors is not None
        np.testing.assert_array_equal(vectors[0], np.array([0.1]))
        mock_log.assert_any_call("의미론적 분할 완료 (1개 조각)", session_id=None)



def test_postprocess_metadata_inheritance():
    """Metadata inheritance from source_docs."""
    source_doc = create_mock_document(
        content="source", 
        metadata={"file_path": "path/to/file.pdf", "file_hash": "hash123", "source": "source_name"}
    )
    split_doc = create_mock_document(content="chunk", metadata={})
    
    _postprocess_metadata([split_doc], source_docs=[source_doc])
    
    assert split_doc.metadata["file_path"] == "path/to/file.pdf"
    assert split_doc.metadata["file_hash"] == "hash123"
    assert split_doc.metadata["source"] == "source_name"

def test_postprocess_metadata_reference_detection():
    """Detection of reference sections."""
    # Test Korean keyword
    doc_ko = create_mock_document(content="## 참고문헌\n[1] source")
    _postprocess_metadata([doc_ko])
    assert doc_ko.metadata["is_reference"] is True
    assert doc_ko.metadata["is_content"] is False

    # Test English keyword
    doc_en = create_mock_document(content="## References\n[1] source")
    _postprocess_metadata([doc_en])
    assert doc_en.metadata["is_reference"] is True
    assert doc_en.metadata["is_content"] is False

    # Test end of reference section
    doc_end = create_mock_document(content="## Appendix\nNext section")
    # We need to simulate being in a reference section
    # Since _postprocess_metadata iterates through docs, we can provide a sequence
    docs = [
        create_mock_document(content="## References\n[1] source"),
        create_mock_document(content="## Appendix\nNext section")
    ]
    _postprocess_metadata(docs)
    assert docs[0].metadata["is_reference"] is True
    assert docs[1].metadata["is_reference"] is False

def test_postprocess_metadata_noise_detection():
    """Detection of noise (DOI, etc.)."""
    doc_noise = create_mock_document(content="This is a DOI: 10.1000/123456 and another DOI: 10.1001/789012 and a third DOI: 10.1002/345678")
    _postprocess_metadata([doc_noise])
    assert doc_noise.metadata["is_content"] is False

    doc_comma = create_mock_document(content="a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, 1, 2, 3")
    _postprocess_metadata([doc_comma])
    assert doc_comma.metadata["is_content"] is False
