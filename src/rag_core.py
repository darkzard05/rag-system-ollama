"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import os
import logging
import hashlib
import json
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
import tempfile
import pickle

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.retrievers import EnsembleRetriever

from config import (
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from session import SessionManager
from utils import log_operation
from graph_builder import build_graph


logger = logging.getLogger(__name__)

# --- 문서 처리 ---
@log_operation("Load PDF documents")
def _load_pdf_docs(pdf_file_bytes: bytes) -> List["Document"]:
    from langchain_community.document_loaders import PyMuPDFLoader

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_file_bytes)
        temp_file_path = temp_file.name

    try:
        loader = PyMuPDFLoader(file_path=temp_file_path)
        return loader.load()
    finally:
        os.remove(temp_file_path)


@log_operation("Split documents")
def _split_documents(docs: List["Document"]) -> List["Document"]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
    )
    return chunker.split_documents(docs)


# --- 캐시 관리 ---
def _serialize_docs(docs: List["Document"]) -> List[Dict]:
    """
    문서 객체 목록을 직렬화 가능한 딕셔너리 목록으로 변환합니다.
    """
    return [doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() for doc in docs]

def _deserialize_docs(docs_as_dicts: List[Dict]) -> List["Document"]:
    """
    직렬화된 딕셔너리 목록을 문서 객체 목록으로 변환합니다.
    """
    from langchain_core.documents import Document

    if hasattr(Document, "model_validate"):
        return [Document.model_validate(d) for d in docs_as_dicts]
    return [Document.parse_obj(d) for d in docs_as_dicts]

class VectorStoreCache:
    """
    벡터 저장소 및 리트리버 캐시를 관리(경로 생성, 저장, 로드)합니다.
    """
    def __init__(self, file_bytes: bytes, embedding_model_name: str):
        (
            self.cache_dir,
            self.doc_splits_path,
            self.faiss_index_path,
            self.bm25_retriever_path,
        ) = self._get_cache_paths(file_bytes, embedding_model_name)

    def _get_cache_paths(
        self, file_bytes: bytes, embedding_model_name: str
    ) -> Tuple[str, str, str, str]:
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        model_name_slug = embedding_model_name.replace("/", "_")
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}"
        )
        doc_splits_path = os.path.join(cache_dir, "doc_splits.json")
        faiss_index_path = os.path.join(cache_dir, "faiss_index")
        bm25_retriever_path = os.path.join(cache_dir, "bm25_retriever.pkl")
        return cache_dir, doc_splits_path, faiss_index_path, bm25_retriever_path

    def load(
        self, embedder: "HuggingFaceEmbeddings"
    ) -> Tuple[
        Optional[List["Document"]], Optional["FAISS"], Optional["BM25Retriever"]
    ]:
        from langchain_community.vectorstores import FAISS

        if not all(
            os.path.exists(p)
            for p in [
                self.doc_splits_path,
                self.faiss_index_path,
                self.bm25_retriever_path,
            ]
        ):
            return None, None, None
        try:
            # 1. 분할된 문서 로드
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_splits = _deserialize_docs(json.load(f))

            # 2. FAISS 벡터 저장소 로드
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            # 3. BM25 리트리버 로드
            with open(self.bm25_retriever_path, "rb") as f:
                bm25_retriever = pickle.load(f)

            logger.info(f"Full RAG cache loaded from '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}. Rebuilding cache.")
            return None, None, None

    def save(
        self,
        doc_splits: List["Document"],
        vector_store: "FAISS",
        bm25_retriever: "BM25Retriever",
    ):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # 1. 분할된 문서 저장
            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(_serialize_docs(doc_splits), f, ensure_ascii=False, indent=4)
            # 2. FAISS 벡터 저장소 저장
            vector_store.save_local(self.faiss_index_path)
            # 3. BM25 리트리버 저장
            with open(self.bm25_retriever_path, "wb") as f:
                pickle.dump(bm25_retriever, f)

            logger.info(f"Full RAG cache saved to '{self.cache_dir}'")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


# --- 리트리버 생성 ---
@log_operation("Create FAISS vector store")
def _create_vector_store(
    docs: List["Document"], embedder: "HuggingFaceEmbeddings"
) -> "FAISS":
    """
    문서 목록과 임베더를 사용하여 FAISS 벡터 저장소를 생성합니다.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, embedder)


@log_operation("Create BM25 retriever")
def _create_bm25_retriever(docs: List["Document"]) -> "BM25Retriever":
    """
    문서 목록을 사용하여 BM25 리트리버를 생성합니다.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever


@log_operation("Create Ensemble retriever")
def _create_ensemble_retriever(
    vector_store: "FAISS", bm25_retriever: "BM25Retriever"
) -> "EnsembleRetriever":
    """
    FAISS 및 BM25 리트리버를 결합한 앙상블 리트리버를 생성합니다.
    """
    from langchain.retrievers import EnsembleRetriever

    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=RETRIEVER_CONFIG["ensemble_weights"],
    )


# --- 파이프라인 구축 ---
@log_operation("Load/Build retrieval components")
def _load_and_build_retrieval_components(
    file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[List["Document"], "FAISS", "BM25Retriever", bool]:
    """
    캐시에서 검색 구성 요소를 로드하거나, 캐시가 없으면 새로 생성하고 저장합니다.
    """
    cache = VectorStoreCache(file_bytes, embedder.model_name)
    doc_splits, vector_store, bm25_retriever = cache.load(embedder)
    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        docs = _load_pdf_docs(file_bytes)
        doc_splits = _split_documents(docs)
        vector_store = _create_vector_store(doc_splits, embedder)
        bm25_retriever = _create_bm25_retriever(doc_splits)
        cache.save(doc_splits, vector_store, bm25_retriever)

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("Build RAG pipeline")
def build_rag_pipeline(
    uploaded_file_name: str, file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[str, bool]:
    """
    RAG 파이프라인을 구축하고 세션에 저장합니다.
    """
    doc_splits, vector_store, bm25_retriever, cache_used = (
        _load_and_build_retrieval_components(file_bytes, embedder)
    )
    final_retriever = _create_ensemble_retriever(vector_store, bm25_retriever)
    rag_app = build_graph(retriever=final_retriever)

    SessionManager.set("processed_document_splits", doc_splits)
    SessionManager.set("qa_chain", rag_app)
    SessionManager.set("pdf_processed", True)

    logger.info(
        f"RAG pipeline built successfully for '{uploaded_file_name}' (Cache used: {cache_used})"
    )

    if cache_used:
        return f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다.", True
    return (
        f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
        "이제 문서 내용에 대해 자유롭게 질문해보세요."
    ), False


@log_operation("Update LLM in pipeline")
def update_llm_in_pipeline(llm):
    """
    세션의 LLM을 교체합니다. 그래프가 세션에서 LLM을 가져오므로 재빌드할 필요가 없습니다.
    """
    if not SessionManager.get("pdf_processed"):
        raise ValueError("RAG 파이프라인이 구축되지 않아 LLM을 업데이트할 수 없습니다.")

    SessionManager.set("llm", llm)
    logger.info(f"Session LLM updated to new model: '{llm.model}'")
