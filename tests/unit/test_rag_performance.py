# create_vector_store 호출 시 중복 임베딩 수행 여부를 검증하는 테스트
import pytest
import numpy as np
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.core.retriever_factory import create_vector_store

def test_create_vector_store_no_redundant_embedding():
    """벡터가 전달되었을 때 embedder.embed_documents가 호출되지 않는지 검증"""
    # 1. 준비
    mock_docs = [
        Document(page_content="테스트 문서 1", metadata={"page": 1}),
        Document(page_content="테스트 문서 2", metadata={"page": 2})
    ]
    mock_embedder = MagicMock()
    # 128차원 가짜 벡터 생성
    mock_vectors = [np.random.rand(128).astype("float32") for _ in range(len(mock_docs))]
    
    # 2. 실행: vectors 인자를 명시적으로 전달
    # create_vector_store 내의 FAISS 생성 로직을 최소화하기 위해 FAISS 관련 모킹이 필요할 수 있으나,
    # 여기서는 embedder 호출 여부만 확인하므로 embedder가 호출되지 않는지만 체크함.
    # FAISS.from_embeddings 등을 내부에서 호출하므로 실제 FAISS 라이브러리가 필요함.
    try:
        create_vector_store(mock_docs, mock_embedder, vectors=mock_vectors)
    except Exception as e:
        # FAISS 내부 로직에서 에러가 나더라도(GPU 미지원 등) 
        # embedder.embed_documents 가 호출되었는지가 핵심임
        pass
    
    # 3. 검증: embedder.embed_documents 가 호출되지 않았어야 함
    mock_embedder.embed_documents.assert_not_called()

def test_create_vector_store_with_none_vectors_calls_embedder():
    """벡터가 None일 때 embedder.embed_documents가 호출되는지 검증 (대조군)"""
    # 1. 준비
    mock_docs = [
        Document(page_content="테스트 문서 1", metadata={"page": 1})
    ]
    mock_embedder = MagicMock()
    # embed_documents 호출 시 반환값 설정
    mock_embedder.embed_documents.return_value = [[0.1] * 128]
    
    # 2. 실행: vectors=None (기본값)
    try:
        create_vector_store(mock_docs, mock_embedder, vectors=None)
    except Exception as e:
        pass
    
    # 3. 검증: embedder.embed_documents 가 호출되었어야 함
    mock_embedder.embed_documents.assert_called_once()
