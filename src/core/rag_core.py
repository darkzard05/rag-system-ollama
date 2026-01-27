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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from common.typing_utils import (
    DocumentList,
    DocumentDict,
    DocumentDictList,
    ConfigDict,
    Embeddings,
    T,
)

import streamlit as st
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

from common.exceptions import (
    PDFProcessingError,
    EmptyPDFError,
    InsufficientChunksError,
    VectorStoreError,
    EmbeddingModelError,
)
from common.config import (
    EMBEDDING_BATCH_SIZE,
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
    CACHE_SECURITY_LEVEL,
    CACHE_HMAC_SECRET,
    CACHE_TRUSTED_PATHS,
    CACHE_VALIDATION_ON_FAILURE,
    CACHE_CHECK_PERMISSIONS,
)
from core.session import SessionManager
from common.utils import log_operation, preprocess_text
from core.graph_builder import build_graph
from security.cache_security import (
    CacheSecurityManager,
    CacheIntegrityError,
    CachePermissionError,
    CacheTrustError,
    CacheSecurityError,
)

from core.semantic_chunker import EmbeddingBasedSemanticChunker

from services.optimization.batch_optimizer import get_optimal_batch_size
from services.optimization.index_optimizer import get_index_optimizer, IndexOptimizationConfig
from services.monitoring.performance_monitor import get_performance_monitor, OperationType

import numpy as np
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

class RAGSystem:
    """
    Backward-compatible facade for older tests/clients.

    The current app path uses `build_rag_pipeline()` + LangGraph, but some tests
    expect a `RAGSystem` class with basic document processing helpers.
    """

    def __init__(self):
        # Keep init lightweight: do not auto-load models or talk to Ollama.
        pass

    def process_documents(self, documents: List[str]) -> None:
        """Validate input documents; raise consistent domain errors."""
        if not documents:
            raise EmptyPDFError(
                filename="(in-memory)",
                details={"reason": "documents list is empty"},
            )

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk raw text documents using the configured splitter."""
        self.process_documents(documents)
        texts = [preprocess_text(t) for t in documents if preprocess_text(t)]
        if not texts:
            raise EmptyPDFError(
                filename="(in-memory)",
                details={"reason": "no usable text after preprocessing"},
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG.get("chunk_size", 500),
            chunk_overlap=TEXT_SPLITTER_CONFIG.get("chunk_overlap", 100),
        )
        chunks: List[str] = []
        for t in texts:
            chunks.extend(splitter.split_text(t))
        return [c for c in chunks if c.strip()]

    def retrieve_documents(self, query: str) -> List[Document]:
        """Placeholder for legacy interface (real retrieval happens in LangGraph)."""
        if not query:
            return []
        return []

    def generate_response(self, query: str) -> str:
        """Placeholder for legacy interface (real generation happens in LangGraph)."""
        if not query:
            return ""
        return "RAGSystem.generate_response is not wired in legacy facade; use the Streamlit app pipeline."


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


def _load_pdf_docs(file_path: str, file_name: str, on_progress=None) -> List[Document]:
    """
    PDF 파일을 디스크에서 로드하여 LangChain Document 객체로 변환합니다.
    """
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        docs = []
        try:
            SessionManager.add_status_log(f"문서 파일 접근 중")
            if on_progress: on_progress()
            
            with fitz.open(file_path) as doc_file:
                total_pages = len(doc_file)
                for page_num, page in enumerate(doc_file):
                    if (page_num + 1) % 5 == 0 or page_num == 0:
                        SessionManager.replace_last_status_log(f"텍스트 추출 중 ({page_num + 1}/{total_pages}p)")
                        if on_progress: on_progress()
                    
                    try:
                        text = page.get_text()
                    except Exception as e:
                        logger.warning(f"페이지 {page_num+1} 텍스트 추출 실패: {e}")
                        text = ""
                    
                    if text:
                        clean_text = preprocess_text(text)
                        if clean_text and len(clean_text) > 10:
                            metadata = {
                                "source": file_name,
                                "page": int(page_num + 1),
                                "total_pages": int(total_pages)
                            }
                            docs.append(Document(page_content=clean_text, metadata=metadata))
            
            SessionManager.replace_last_status_log(f"텍스트 {len(docs)}개 구간 확보")
            logger.info(f"PDF 로드 완료: {len(docs)}/{total_pages} 페이지 추출됨.")
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"PDF 로드 중 오류 발생: {e}")
            op.error = str(e)
            raise


def _split_documents(
    docs: List[Document],
    embedder: Optional["HuggingFaceEmbeddings"] = None,
) -> List[Document]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    """
    SessionManager.add_status_log("의미 단위 문장 분할 중")
    with monitor.track_operation(OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}) as op:
        use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)

        if use_semantic and embedder:
            # 배치 사이즈 결정 로직 개선
            if isinstance(EMBEDDING_BATCH_SIZE, int):
                batch_size = EMBEDDING_BATCH_SIZE
            elif str(EMBEDDING_BATCH_SIZE).lower() == "auto":
                batch_size = get_optimal_batch_size(model_type="embedding")
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

        op.tokens = sum(len(doc.page_content.split()) for doc in split_docs)
        return split_docs


def _serialize_docs(docs: DocumentList) -> DocumentDictList:
    return [
        doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() for doc in docs
    ]


def _deserialize_docs(docs_as_dicts: DocumentDictList) -> DocumentList:
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
    """벡터 저장소 및 리트리버 캐시 관리자"""

    def __init__(self, file_path: str, embedding_model_name: str):
        self.cache_dir, self.doc_splits_path, self.faiss_index_path, self.bm25_retriever_path = self._get_cache_paths(
            file_path, embedding_model_name
        )
        
        # 캐시 보안 관리자 초기화
        self.security_manager = CacheSecurityManager(
            security_level=CACHE_SECURITY_LEVEL,
            hmac_secret=CACHE_HMAC_SECRET,
            trusted_paths=CACHE_TRUSTED_PATHS,
            check_permissions=CACHE_CHECK_PERMISSIONS,
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
        """
        캐시된 RAG 컴포넌트를 로드합니다.
        
        보안 검증:
        - 파일 무결성 (SHA256)
        - 파일 권한
        - HMAC 서명 (high 레벨)
        
        Returns:
            (문서 리스트, FAISS 벡터 저장소, BM25 리트리버) 또는 (None, None, None)
        """
        if not all(
            os.path.exists(p)
            for p in [self.doc_splits_path, self.faiss_index_path, self.bm25_retriever_path]
        ):
            logger.debug(f"캐시 파일 누락: {self.cache_dir}")
            return None, None, None

        try:
            # 1. 문서 로드 (JSON - 이미 안전함)
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_splits = _deserialize_docs(json.load(f))

            # 2. FAISS 로드
            # 보안: allow_dangerous_deserialization=True + 무결성 검증
            try:
                self.security_manager.verify_cache_trust(self.faiss_index_path)
                logger.debug(f"FAISS 캐시 신뢰 검증: {self.faiss_index_path}")
            except CacheTrustError as e:
                if CACHE_VALIDATION_ON_FAILURE == "fail":
                    raise
                logger.warning(f"FAISS 신뢰 검증 경고: {e}. 계속 진행합니다.")
            
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            # 3. BM25 로드 (Pickle)
            # 보안 강화: 무결성 검증 + 권한 검사
            bm25_metadata_path = self.bm25_retriever_path + ".meta"
            
            try:
                # 무결성 검증 수행
                self.security_manager.verify_cache_integrity(
                    self.bm25_retriever_path,
                    metadata_path=bm25_metadata_path,
                )
                logger.debug(f"BM25 무결성 검증 성공: {self.bm25_retriever_path}")
            except CacheIntegrityError as e:
                error_msg = f"BM25 캐시 무결성 검증 실패: {e}"
                logger.error(error_msg)
                
                if CACHE_VALIDATION_ON_FAILURE == "regenerate":
                    logger.info("캐시를 재생성합니다.")
                    return None, None, None
                elif CACHE_VALIDATION_ON_FAILURE == "fail":
                    raise
                else:  # warn
                    logger.warning(error_msg)
            except CachePermissionError as e:
                error_msg = f"BM25 캐시 권한 오류: {e}"
                logger.warning(error_msg)
                if CACHE_VALIDATION_ON_FAILURE == "fail":
                    raise
            except CacheTrustError as e:
                error_msg = f"BM25 캐시 신뢰 검증 실패: {e}"
                logger.warning(error_msg)
                if CACHE_VALIDATION_ON_FAILURE == "fail":
                    raise
            
            # 검증 통과 후 로드
            with open(self.bm25_retriever_path, "rb") as f:
                bm25_retriever = pickle.load(f)

            logger.info(f"RAG 캐시 로드 완료: '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever

        except Exception as e:
            logger.warning(f"캐시 로드 실패 (손상 가능성 또는 버전 불일치): {e}. 캐시를 재생성합니다.")
            return None, None, None

    def save(
        self,
        doc_splits: DocumentList,
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
    ) -> None:
        """
        RAG 컴포넌트를 캐시에 저장합니다.
        
        보안 처리:
        - 메타데이터 생성 (SHA256 해시 포함)
        - HMAC 서명 (high 레벨)
        - 파일 권한 설정
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # 1. 문서 저장
            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(_serialize_docs(doc_splits), f, ensure_ascii=False, indent=4)
            logger.debug(f"문서 splits 저장: {self.doc_splits_path}")

            # 2. FAISS 저장
            vector_store.save_local(self.faiss_index_path)
            logger.debug(f"FAISS 벡터 저장소 저장: {self.faiss_index_path}")

            # 3. BM25 저장 (Pickle) + 메타데이터
            with open(self.bm25_retriever_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.debug(f"BM25 리트리버 저장: {self.bm25_retriever_path}")
            
            # 메타데이터 생성 및 저장
            try:
                metadata = self.security_manager.create_metadata_for_file(
                    self.bm25_retriever_path,
                    description=f"BM25 retriever cache"
                )
                
                # high 레벨: HMAC 서명 추가
                if CACHE_SECURITY_LEVEL == "high" and CACHE_HMAC_SECRET:
                    try:
                        with open(self.bm25_retriever_path, "rb") as f:
                            file_data = f.read()
                        metadata.integrity_hmac = self.security_manager.compute_integrity_hmac(file_data)
                        logger.debug("HMAC 서명 생성 완료")
                    except Exception as e:
                        logger.warning(f"HMAC 서명 생성 실패: {e}")
                
                bm25_metadata_path = self.bm25_retriever_path + ".meta"
                self.security_manager.save_cache_metadata(bm25_metadata_path, metadata)
                logger.info(f"캐시 메타데이터 저장: {bm25_metadata_path}")
            except Exception as e:
                logger.warning(f"메타데이터 저장 실패: {e}. 캐시는 저장되었지만 검증 불가능합니다.")

            logger.info(f"RAG 캐시 저장 완료: '{self.cache_dir}'")
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            raise


@log_operation("FAISS 벡터 저장소 생성")
def _create_vector_store(
    docs: List[Document],
    embedder: "HuggingFaceEmbeddings",
    vectors: Optional[List[np.ndarray]] = None,
) -> FAISS:
    """
    FAISS 벡터 저장소를 생성합니다. 
    이미 계산된 벡터(embeddings)가 있으면 재사용하여 성능을 최적화합니다.
    """
    if vectors is not None:
        # 텍스트와 임베딩 쌍으로 생성 (임베딩 모델 재호출 방지)
        text_embeddings = zip([d.page_content for d in docs], vectors)
        return FAISS.from_embeddings(text_embeddings, embedder, metadatas=[d.metadata for d in docs])
    
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
    _on_progress=None
) -> Tuple[DocumentList, FAISS, BM25Retriever, bool]:

    cache = VectorStoreCache(file_path, embedding_model_name)
    doc_splits, vector_store, bm25_retriever = cache.load(_embedder)

    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        docs = _load_pdf_docs(file_path, file_name, on_progress=_on_progress)
        # 빈 문서 처리 추가
        if not docs:
            raise EmptyPDFError(
                filename=file_name,
                details={"reason": "PDF에서 텍스트를 추출할 수 없음 (이미지 위주 또는 보호된 파일)"}
            )

        if _on_progress: _on_progress()
        doc_splits = _split_documents(docs, _embedder)
        if _on_progress: _on_progress()

        if not doc_splits:
            raise InsufficientChunksError(
                chunk_count=0,
                min_required=1,
                details={"reason": "청킹 후 유효한 청크가 없음"}
            )

        # [최적화] 인덱스 최적화 적용 및 임베딩 재사용
        optimized_vectors = None
        try:
            SessionManager.add_status_log("인덱스 최적화 중")
            if on_progress: on_progress()
            
            # 임시 벡터 생성 (최적화 분석용)
            texts = [d.page_content for d in doc_splits]
            # 여기서 한 번 임베딩을 수행함
            vectors = _embedder.embed_documents(texts)
            vectors_np = [np.array(v) for v in vectors]
            
            optimizer = get_index_optimizer()
            # 최적화된 문서와 '벡터'를 함께 반환받음
            optimized_docs, optimized_vectors, stats = optimizer.optimize_index(doc_splits, vectors_np)
            
            logger.info(
                f"인덱스 최적화 완료: {len(doc_splits)} -> {len(optimized_docs)} 청크 "
                f"(프루닝: {stats.pruned_documents})"
            )
            SessionManager.replace_last_status_log(f"중복 내용 {stats.pruned_documents}개 정리")
            if on_progress: on_progress()
            
            doc_splits = optimized_docs
            # FAISS에서 사용할 수 있도록 리스트 형태로 유지
        except Exception as e:
            logger.warning(f"인덱스 최적화 실패 (계속 진행): {e}")
            optimized_vectors = None

        # 최적화된 벡터가 있으면 재사용, 없으면 내부에서 새로 생성
        vector_store = _create_vector_store(doc_splits, _embedder, vectors=optimized_vectors)
        bm25_retriever = _create_bm25_retriever(doc_splits)
        cache.save(doc_splits, vector_store, bm25_retriever)
        SessionManager.add_status_log("신규 인덱싱 완료") # 추가

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str,
    file_path: str,
    embedder: "HuggingFaceEmbeddings",
    on_progress=None
) -> Tuple[str, bool]:
    """
    RAG 파이프라인을 구축하고 세션에 저장합니다.
    """
    # [최적화] embedder 객체는 해싱에서 제외(_)하고, 모델명을 명시적 키로 전달
    doc_splits, vector_store, bm25_retriever, cache_used = (
        _load_and_build_retrieval_components(
            file_path, 
            uploaded_file_name, 
            _embedder=embedder, 
            embedding_model_name=embedder.model_name,
            _on_progress=on_progress
        )
    )
    
    if cache_used:
        SessionManager.add_status_log("캐시 데이터 로드") 
        if on_progress: on_progress()

    final_retriever = _create_ensemble_retriever(vector_store, bm25_retriever)
    rag_engine = build_graph(retriever=final_retriever)

    SessionManager.set("vector_store", vector_store)
    SessionManager.set("rag_engine", rag_engine)
    SessionManager.set("pdf_processed", True)
    SessionManager.add_status_log("질문 가능")
    if on_progress: on_progress()

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
def update_llm_in_pipeline(llm: Optional[T]) -> None:
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
