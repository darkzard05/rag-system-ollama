"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import os
import logging
import hashlib
import json
import functools
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
import tempfile
import pickle

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
)
from session import SessionManager
from utils import log_operation, preprocess_text


logger = logging.getLogger(__name__)


def _load_pdf_docs(pdf_file_bytes: bytes) -> List["Document"]:
    """
    PDF 파일 바이트를 받아서 LangChain Document 객체 목록으로 로드합니다.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_file_bytes)
        temp_file_path = temp_file.name

    try:
        loader = PyMuPDFLoader(file_path=temp_file_path)
        docs = loader.load()
        
        # 텍스트 전처리 적용
        for doc in docs:
            doc.page_content = preprocess_text(doc.page_content)
            
        return docs
    finally:
        os.remove(temp_file_path)


def _split_documents(docs: List["Document"], embedder=None) -> List["Document"]:
    """
    문서 목록을 텍스트 분할기를 사용하여 분할합니다.
    
    의미론적 분할이 활성화되면 임베딩 기반 의미 분할기를 사용하고,
    그렇지 않으면 기존의 RecursiveCharacterTextSplitter를 사용합니다.
    
    Args:
        docs: 분할할 문서 리스트
        embedder: 의미론적 분할을 위한 임베딩 모델 (선택사항)
        
    Returns:
        분할된 문서 리스트
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    
    # 의미론적 분할 활성화 여부 확인
    use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)
    
    if use_semantic and embedder:
        # 의미론적 분할기 사용
        from semantic_chunker import EmbeddingBasedSemanticChunker
        
        # 배치 사이즈 결정
        try:
            batch_size = int(EMBEDDING_BATCH_SIZE)
        except (ValueError, TypeError):
            batch_size = 64  # 'auto'이거나 잘못된 값일 경우 기본값
            
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
        
        # 문서 전체를 통합하여 의미론적 분할 (페이지 경계 극복)
        split_docs = semantic_chunker.split_documents(docs)
        
        logger.info(
            f"문서 분할 완료: {len(docs)}개 문서 -> {len(split_docs)}개 의미론적 청크"
        )
        return split_docs
    else:
        # 기존의 RecursiveCharacterTextSplitter 사용
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
        )
        split_docs = chunker.split_documents(docs)

    # 모든 청크에 인덱스 부여 (문서 병합 및 순서 복원용)
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_index"] = i

    return split_docs


def _serialize_docs(docs: List["Document"]) -> List[Dict]:
    """
    문서 객체 목록을 직렬화 가능한 딕셔너리 목록으로 변환합니다.

    Args:
        docs (List[Document]): 문서 객체 목록.

    Returns:
        List[Dict]: 직렬화된 문서 딕셔너리 목록.
    """
    return [doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() for doc in docs]

def _deserialize_docs(docs_as_dicts: List[Dict]) -> List["Document"]:
    """
    직렬화된 딕셔너리 목록을 문서 객체 목록으로 변환합니다.

    Args:
        docs_as_dicts (List[Dict]): 직렬화된 문서 딕셔너리 목록.

    Returns:
        List[Document]: 변환된 문서 객체 목록.
    """
    from langchain_core.documents import Document

    return [Document(**d) for d in docs_as_dicts]

@functools.lru_cache(maxsize=1)
def _compute_config_hash() -> str:
    """
    설정(의미분할, 리트리버)이 변경되었는지 감지하기 위한 해시 생성.
    설정이 변경되면 다른 캐시 키가 생성되어 이전 캐시를 무효화합니다.
    """
    config_dict = {
        "semantic_chunker": SEMANTIC_CHUNKER_CONFIG,
        "text_splitter": TEXT_SPLITTER_CONFIG,
        "retriever": RETRIEVER_CONFIG,
    }
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class VectorStoreCache:
    """
    벡터 저장소 및 리트리버 캐시를 관리합니다.

    캐시 경로 생성, 저장소/리트리버 저장 및 로드 기능을 제공합니다.
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
        config_hash = _compute_config_hash()  # 설정 기반 해시 추가
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}_{config_hash}"
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
        """
        캐시에서 검색 구성 요소를 로드합니다.

        Args:
            embedder (HuggingFaceEmbeddings): 임베딩 모델.

        Returns:
            Tuple: (분할된 문서, FAISS 벡터 저장소, BM25 리트리버) 또는 모두 None.
        """
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
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_splits = _deserialize_docs(json.load(f))

            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            with open(self.bm25_retriever_path, "rb") as f:
                bm25_retriever = pickle.load(f)

            logger.info(f"RAG 캐시 로드 완료: '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}. 캐시를 재생성합니다.")
            return None, None, None

    def save(
        self,
        doc_splits: List["Document"],
        vector_store: "FAISS",
        bm25_retriever: "BM25Retriever",
    ):
        """
        검색 구성 요소를 캐시에 저장합니다.

        Args:
            doc_splits (List[Document]): 분할된 문서 목록.
            vector_store (FAISS): FAISS 벡터 저장소.
            bm25_retriever (BM25Retriever): BM25 리트리버.
        """
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
    docs: List["Document"], embedder: "HuggingFaceEmbeddings"
) -> "FAISS":
    """
    문서 목록과 임베더를 사용하여 FAISS 벡터 저장소를 생성합니다.

    Args:
        docs (List[Document]): 문서 목록.
        embedder (HuggingFaceEmbeddings): 임베딩 모델.

    Returns:
        FAISS: 생성된 FAISS 벡터 저장소.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, embedder)


def _create_bm25_retriever(docs: List["Document"]) -> "BM25Retriever":
    """
    문서 목록을 사용하여 BM25 리트리버를 생성합니다.

    Args:
        docs (List[Document]): 문서 목록.

    Returns:
        BM25Retriever: 생성된 BM25 리트리버.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever


def _create_ensemble_retriever(
    vector_store: "FAISS", bm25_retriever: "BM25Retriever"
) -> "EnsembleRetriever":
    """
    FAISS 및 BM25 리트리버를 결합한 앙상블 리트리버를 생성합니다.

    Args:
        vector_store (FAISS): FAISS 벡터 저장소.
        bm25_retriever (BM25Retriever): BM25 리트리버.

    Returns:
        EnsembleRetriever: 생성된 앙상블 리트리버.
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
@log_operation("검색 컴포넌트 로드/생성")
def _load_and_build_retrieval_components(
    file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[List["Document"], "FAISS", "BM25Retriever", bool]:
    """
    캐시에서 검색 구성 요소를 로드하거나, 캐시가 없으면 새로 생성하고 저장합니다.

    Args:
        file_bytes (bytes): PDF 파일의 바이트 데이터.
        embedder (HuggingFaceEmbeddings): 임베딩 모델.

    Returns:
        Tuple: (분할된 문서, FAISS 저장소, BM25 리트리버, 캐시 사용 여부).
    """
    cache = VectorStoreCache(file_bytes, embedder.model_name)
    doc_splits, vector_store, bm25_retriever = cache.load(embedder)
    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        docs = _load_pdf_docs(file_bytes)
        doc_splits = _split_documents(docs, embedder)

        if not doc_splits:
            raise ValueError(
                "PDF에서 텍스트를 추출할 수 없습니다. 스캔된 문서이거나 텍스트가 없는 파일일 수 있습니다."
            )

        vector_store = _create_vector_store(doc_splits, embedder)
        bm25_retriever = _create_bm25_retriever(doc_splits)
        cache.save(doc_splits, vector_store, bm25_retriever)

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str, file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[str, bool]:
    """
    RAG 파이프라인을 구축하고 세션에 저장합니다.

    Args:
        uploaded_file_name (str): 업로드된 파일의 이름.
        file_bytes (bytes): 파일의 바이트 데이터.
        embedder (HuggingFaceEmbeddings): 임베딩 모델.

    Returns:
        Tuple[str, bool]: (성공 메시지, 캐시 사용 여부).
    """
    from graph_builder import build_graph

    doc_splits, vector_store, bm25_retriever, cache_used = (
        _load_and_build_retrieval_components(file_bytes, embedder)
    )
    final_retriever = _create_ensemble_retriever(vector_store, bm25_retriever)
    rag_app = build_graph(retriever=final_retriever)

    SessionManager.set("processed_document_splits", doc_splits)
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
def update_llm_in_pipeline(llm):
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
    logger.info(f"세션 LLM 업데이트 완료: '{llm.model}'")
