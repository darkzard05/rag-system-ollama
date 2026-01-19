"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import functools
import hashlib
import json
import logging
import os
import pickle
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import streamlit as st
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_BATCH_SIZE,
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from session import SessionManager
from utils import log_operation, preprocess_text
from graph_builder import build_graph

from semantic_chunker import EmbeddingBasedSemanticChunker

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: str) -> str:
    """파일의 SHA256 해시를 계산합니다."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"파일 해시 계산 실패: {e}")
        return ""


def _load_pdf_docs(file_path: str, file_name: str) -> List[Document]:
    """
    PDF 파일을 디스크에서 로드하여 LangChain Document 객체로 변환합니다.
    [최적화] 메모리 효율성을 위해 파일 경로를 사용합니다.
    """
    docs = []
    try:
        # 파일 경로에서 PDF 열기
        with fitz.open(file_path) as doc_file:
            total_pages = len(doc_file)
            for page_num, page in enumerate(doc_file):
                try:
                    text = page.get_text()
                except Exception as e:
                    logger.warning(f"페이지 {page_num+1} 텍스트 추출 실패: {e}")
                    text = ""
                
                # [강화] 텍스트 전처리 및 최소 길이 검사
                if text:
                    clean_text = preprocess_text(text)
                    # 너무 짧은 페이지는 건너뛰되, 로그로 남김
                    if clean_text and len(clean_text) > 10:
                        metadata = {
                            "source": file_name,
                            "page": int(page_num + 1),  # 명시적 int 변환
                            "total_pages": int(total_pages)
                        }
                        docs.append(Document(page_content=clean_text, metadata=metadata))
        
        logger.info(f"PDF 로드 완료: {len(docs)}/{total_pages} 페이지 추출됨.")
        return docs

    except Exception as e:
        logger.error(f"PDF 로드 중 오류 발생: {e}")
        raise


def _split_documents(
    docs: List[Document],
    embedder: Optional["HuggingFaceEmbeddings"] = None,
) -> List[Document]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    """
    use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)

    if use_semantic and embedder:
        # 배치 사이즈 결정 로직 개선
        if isinstance(EMBEDDING_BATCH_SIZE, int):
            batch_size = EMBEDDING_BATCH_SIZE
        elif str(EMBEDDING_BATCH_SIZE).lower() == "auto":
            batch_size = 64  # auto일 경우 현재는 기본값 64 사용 (확장 가능)
        else:
            try:
                batch_size = int(EMBEDDING_BATCH_SIZE)
            except (ValueError, TypeError):
                batch_size = 64

        semantic_chunker = EmbeddingBasedSemanticChunker(
            embedder=embedder,
            breakpoint_threshold_type=SEMANTIC_CHUNKER_CONFIG.get(
                "breakpoint_threshold_type", "percentile"
            ),
            breakpoint_threshold_value=float(
                SEMANTIC_CHUNKER_CONFIG.get("breakpoint_threshold_value", 95.0)
            ),
            sentence_split_regex=SEMANTIC_CHUNKER_CONFIG.get(
                "sentence_split_regex", r"[.!?]\s+"
            ),
            min_chunk_size=int(SEMANTIC_CHUNKER_CONFIG.get("min_chunk_size", 100)),
            max_chunk_size=int(SEMANTIC_CHUNKER_CONFIG.get("max_chunk_size", 800)),
            similarity_threshold=float(
                SEMANTIC_CHUNKER_CONFIG.get("similarity_threshold", 0.5)
            ),
            batch_size=batch_size,
        )

        split_docs = semantic_chunker.split_documents(docs)
        logger.info(f"의미론적 분할 완료: {len(docs)} 문서 -> {len(split_docs)} 청크")
    else:
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
        )
        split_docs = chunker.split_documents(docs)
        logger.info(f"기본 분할 완료: {len(docs)} 문서 -> {len(split_docs)} 청크")

    # 청크 인덱스 메타데이터 추가
    for i, doc in enumerate(split_docs):
        doc.metadata = doc.metadata.copy()
        doc.metadata["chunk_index"] = i

    return split_docs


def _serialize_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() for doc in docs
    ]


def _deserialize_docs(docs_as_dicts: List[Dict[str, Any]]) -> List[Document]:
    return [Document(**d) for d in docs_as_dicts]


@functools.lru_cache(maxsize=1)
def _compute_config_hash() -> str:
    """설정 변경 감지용 해시 생성"""
    config_dict = {
        "semantic_chunker": SEMANTIC_CHUNKER_CONFIG,
        "text_splitter": TEXT_SPLITTER_CONFIG,
        "retriever": RETRIEVER_CONFIG,
    }
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]  # 해시 길이 12자리로 확장


class VectorStoreCache:
    """벡터 저장소 및 리트리버 캐시 관리자""" #

    def __init__(self, file_path: str, embedding_model_name: str):
        self.cache_dir, self.doc_splits_path, self.faiss_index_path, self.bm25_retriever_path = self._get_cache_paths(
            file_path, embedding_model_name
        )

    def _get_cache_paths(
        self, file_path: str, embedding_model_name: str
    ) -> Tuple[str, str, str, str]:
        file_hash = _compute_file_hash(file_path)
        # 파일 경로에 안전하지 않은 문자 제거
        model_name_slug = embedding_model_name.replace("/", "_").replace("\\", "_")
        config_hash = _compute_config_hash()

        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}_{config_hash}"
        )

        return (
            cache_dir,
            os.path.join(cache_dir, "doc_splits.json"),
            os.path.join(cache_dir, "faiss_index"),
            os.path.join(cache_dir, "bm25_retriever.pkl"),
        )

    def load(
        self,
        embedder: "HuggingFaceEmbeddings",
    ) -> Tuple[Optional[List[Document]], Optional[FAISS], Optional[BM25Retriever]]:
        if not all(
            os.path.exists(p)
            for p in [self.doc_splits_path, self.faiss_index_path, self.bm25_retriever_path]
        ):
            return None, None, None

        try:
            # 1. 문서 로드 (JSON)
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_splits = _deserialize_docs(json.load(f))

            # 2. FAISS 로드
            # 보안 경고: allow_dangerous_deserialization=True는 신뢰할 수 있는 로컬 파일에만 사용해야 합니다.
            # 이 프로젝트에서는 로컬에서 생성된 캐시만 로드한다고 가정합니다.
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            # 3. BM25 로드 (Pickle)
            # 보안 경고: pickle은 신뢰할 수 없는 데이터에 대해 안전하지 않습니다.
            with open(self.bm25_retriever_path, "rb") as f:
                bm25_retriever = pickle.load(f)

            logger.info(f"RAG 캐시 로드 완료: '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever

        except Exception as e:
            logger.warning(f"캐시 로드 실패 (손상 가능성): {e}. 캐시를 재생성합니다.")
            return None, None, None

    def save(
        self,
        doc_splits: List[Document],
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
    ):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(_serialize_docs(doc_splits), f, ensure_ascii=False, indent=4)

            vector_store.save_local(self.faiss_index_path)

            with open(self.bm25_retriever_path, "wb") as f:
                pickle.dump(bm25_retriever, f)

            logger.info(f"RAG 캐시 저장 완료: '{self.cache_dir}'")
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")


@log_operation("FAISS 벡터 저장소 생성")
def _create_vector_store(
    docs: List[Document],
    embedder: "HuggingFaceEmbeddings",
) -> FAISS:
    return FAISS.from_documents(docs, embedder)


def _create_bm25_retriever(docs: List[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever


def _create_ensemble_retriever(
    vector_store: FAISS,
    bm25_retriever: BM25Retriever,
) -> EnsembleRetriever:
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=RETRIEVER_CONFIG["ensemble_weights"],
    )


@log_operation("검색 컴포넌트 로드/생성")
@st.cache_resource(show_spinner=False)
def _load_and_build_retrieval_components(
    file_path: str,
    file_name: str,
    _embedder: "HuggingFaceEmbeddings",
    embedding_model_name: str,
) -> Tuple[List[Document], FAISS, BM25Retriever, bool]:

    cache = VectorStoreCache(file_path, embedding_model_name)
    doc_splits, vector_store, bm25_retriever = cache.load(_embedder)

    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        docs = _load_pdf_docs(file_path, file_name)
        # 빈 문서 처리 추가
        if not docs:
             raise ValueError("PDF에서 내용을 읽을 수 없습니다. (빈 파일 또는 이미지 위주)")

        doc_splits = _split_documents(docs, _embedder)

        if not doc_splits:
            raise ValueError(
                "PDF에서 텍스트를 추출할 수 없습니다. 스캔된 문서이거나 텍스트가 없는 파일일 수 있습니다."
            )

        vector_store = _create_vector_store(doc_splits, _embedder)
        bm25_retriever = _create_bm25_retriever(doc_splits)
        cache.save(doc_splits, vector_store, bm25_retriever)

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str,
    file_path: str,
    embedder: "HuggingFaceEmbeddings",
) -> Tuple[str, bool]:
    """
    RAG 파이프라인을 구축하고 세션에 저장합니다.

    Args:
        uploaded_file_name (str): 업로드된 파일의 이름.
        file_path (str): 업로드된 파일의 임시 경로.
        embedder (HuggingFaceEmbeddings): 임베딩 모델.

    Returns:
        Tuple[str, bool]: (성공 메시지, 캐시 사용 여부).
    """
    # [최적화] embedder 객체는 해싱에서 제외(_)하고, 모델명을 명시적 키로 전달
    doc_splits, vector_store, bm25_retriever, cache_used = (
        _load_and_build_retrieval_components(
            file_path, 
            uploaded_file_name, 
            _embedder=embedder, 
            embedding_model_name=embedder.model_name
        )
    )
    final_retriever = _create_ensemble_retriever(vector_store, bm25_retriever)
    rag_app = build_graph(retriever=final_retriever)

    # SessionManager.set("processed_document_splits", doc_splits) # [메모리 최적화] 불필요한 원본 데이터 저장 제거
    SessionManager.set("vector_store", vector_store)
    SessionManager.set("qa_chain", rag_app)
    SessionManager.set("pdf_processed", True)

    logger.info(
        f"RAG 파이프라인 구축 완료: '{uploaded_file_name}' (캐시 사용: {cache_used})"
    )

    if cache_used:
        return f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다.", True
    return (
        f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
        "이제 문서 내용에 대해 자유롭게 질문해보세요."
    ), False


@log_operation("파이프라인 LLM 업데이트")
def update_llm_in_pipeline(llm: Any) -> None:
    """
    세션의 LLM을 교체합니다.

    그래프가 세션에서 LLM을 가져오므로 재빌드할 필요가 없습니다.

    Args:
        llm: 업데이트할 새로운 LLM 모델.

    Raises:
        ValueError: RAG 파이프라인이 구축되지 않았을 때.
    """
    if not SessionManager.get("pdf_processed"):
        raise ValueError("RAG 파이프라인이 구축되지 않아 LLM을 업데이트할 수 없습니다.")

    SessionManager.set("llm", llm)
    logger.info(f"세션 LLM 업데이트 완료: '{getattr(llm, 'model', 'unknown')}'")
