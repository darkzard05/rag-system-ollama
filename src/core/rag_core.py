"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import pickle
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from common.typing_utils import (
    DocumentDictList,
    DocumentList,
    T,
)

if TYPE_CHECKING:
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

from common.config import (
    CACHE_CHECK_PERMISSIONS,
    CACHE_HMAC_SECRET,
    CACHE_SECURITY_LEVEL,
    CACHE_TRUSTED_PATHS,
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from common.exceptions import (
    EmptyPDFError,
    InsufficientChunksError,
    PDFProcessingError,
    VectorStoreError,
)
from common.utils import log_operation, preprocess_text
from core.graph_builder import build_graph
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.session import SessionManager
from security.cache_security import (
    CacheIntegrityError,
    CachePermissionError,
    CacheSecurityManager,
    CacheTrustError,
)
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization.index_optimizer import get_index_optimizer

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


def bm25_tokenizer(text: str) -> list[str]:
    """BM25용 경량 토크나이저 (Pickle 지원을 위해 모듈 레벨에 정의)"""
    return text.lower().split()


class RAGSystem:
    """
    RAG 시스템의 통합 엔트리포인트 클래스.
    세션 기반 상태 관리와 LangGraph 기반 파이프라인을 연결합니다.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        SessionManager.init_session(session_id=session_id)

    def _ensure_session_context(self) -> None:
        """현재 스레드의 세션 컨텍스트를 보장합니다."""
        SessionManager.set_session_id(self.session_id)

    async def load_document(
        self, file_path: str, file_name: str, embedder: HuggingFaceEmbeddings
    ) -> tuple[str, bool]:
        """
        문서를 로드하고 인덱싱 파이프라인을 실행합니다.

        Args:
            file_path: PDF 파일의 로컬 경로
            file_name: 사용자에게 표시될 파일 이름
            embedder: 사용할 임베딩 모델 인스턴스

        Returns:
            Tuple[성공 메시지, 캐시 사용 여부]
        """
        self._ensure_session_context()
        return build_rag_pipeline(
            uploaded_file_name=file_name, file_path=file_path, embedder=embedder
        )

    async def aquery(self, query: str, llm: T | None = None) -> dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.

        Args:
            query: 사용자 질문
            llm: 사용할 LLM 인스턴스 (생략 시 세션에 저장된 모델 사용)

        Returns:
            GraphOutput 구조의 결과 딕셔너리
        """
        self._ensure_session_context()

        if llm:
            SessionManager.set("llm", llm)

        rag_engine = SessionManager.get("rag_engine")
        if not rag_engine:
            raise VectorStoreError(
                details={
                    "reason": "RAG 엔진이 초기화되지 않았습니다. 문서를 먼저 로드하세요."
                }
            )

        current_llm = SessionManager.get("llm")
        config = {"configurable": {"llm": current_llm}}

        # LangGraph 호출
        return await rag_engine.ainvoke({"input": query}, config=config)

    def get_status(self) -> list[str]:
        """현재 세션의 작업 로그를 가져옵니다."""
        self._ensure_session_context()
        return SessionManager.get("status_logs", [])

    def clear_session(self) -> None:
        """세션 데이터를 초기화합니다."""
        self._ensure_session_context()
        SessionManager.reset_all_state()

    def process_documents(self, documents: list[str]) -> None:
        """입력 문서 리스트를 검증합니다. (테스트 호환성 유지)"""
        if not documents:
            raise EmptyPDFError(
                filename="(in-memory)",
                details={"reason": "documents list is empty"},
            )


def _compute_file_hash(file_path: str, data: bytes | None = None) -> str:
    """
    파일 또는 데이터의 SHA256 해시를 계산합니다.
    [최적화] 데이터가 이미 메모리에 있다면 파일을 다시 읽지 않습니다.
    """
    sha256_hash = hashlib.sha256()
    try:
        if data is not None:
            sha256_hash.update(data)
        else:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"해시 계산 실패: {e}")
        return ""


import concurrent.futures  # noqa: E402


def _extract_page_worker(
    file_path: str, page_num: int, total_pages: int, file_name: str
) -> Document | None:
    """개별 페이지에서 텍스트를 추출하는 워커 함수 (스레드 세이프)"""
    try:
        import fitz  # PyMuPDF
        from langchain_core.documents import Document

        # 각 스레드에서 파일을 새로 열어 독립적인 문서 객체 사용
        with fitz.open(file_path) as doc:
            page = doc[page_num]
            text = page.get_text()
            if text:
                clean_text = preprocess_text(text)
                if clean_text and len(clean_text) > 10:
                    metadata = {
                        "source": file_name,
                        "page": int(page_num + 1),
                        "total_pages": int(total_pages),
                    }
                    return Document(page_content=clean_text, metadata=metadata)
    except Exception as e:
        logger.warning(f"페이지 {page_num + 1} 추출 실패: {e}")
    return None


def _extract_pages_batch_worker(
    file_bytes: bytes, page_range: list[int], total_pages: int, file_name: str
) -> list[tuple[int, Document]]:
    """페이지 범위를 배치로 처리하는 워커 함수 (memoryview 사용으로 복사 방지)"""
    results = []
    try:
        import fitz  # PyMuPDF
        from langchain_core.documents import Document

        # [최적화] memoryview를 사용하여 원본 바이트에 대한 zero-copy 접근 보장
        mv = memoryview(file_bytes)
        with fitz.open(stream=mv, filetype="pdf") as doc:
            for page_num in page_range:
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    if text:
                        clean_text = preprocess_text(text)
                        if clean_text and len(clean_text) > 10:
                            metadata = {
                                "source": file_name,
                                "page": int(page_num + 1),
                                "total_pages": int(total_pages),
                            }
                            results.append(
                                (
                                    page_num,
                                    Document(
                                        page_content=clean_text, metadata=metadata
                                    ),
                                )
                            )
                except Exception as e:
                    logger.warning(f"페이지 {page_num + 1} 추출 실패: {e}")
    except Exception as e:
        logger.error(f"배치 처리 중 문서 오픈 실패: {e}")
    return results


def _load_pdf_docs(
    file_path: str, file_name: str, on_progress: Callable[[], None] | None = None
) -> list[Document]:
    """
    PDF 파일을 메모리에 버퍼링한 후 병렬 배치 처리를 통해 최고 속도로 추출합니다.
    """
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            import fitz  # PyMuPDF

            SessionManager.add_status_log("문서 분석 준비 중")
            if on_progress:
                on_progress()

            # 1. 파일 전체를 메모리에 한 번만 로드 (가장 빠른 방식)
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                # 메타데이터 확인을 위해 임시 오픈
                with fitz.open(stream=file_bytes, filetype="pdf") as doc_file:
                    total_pages = len(doc_file)
            except Exception as e:
                raise PDFProcessingError(
                    filename=file_name,
                    details={"reason": f"파일을 읽을 수 없음: {str(e)}"},
                ) from e

            if total_pages == 0:
                raise EmptyPDFError(
                    filename=file_name, details={"reason": "PDF 페이지가 없습니다."}
                )

            # [최적화] 페이지 수에 따라 동적으로 병렬화 결정
            if total_pages <= 3:
                docs = []
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    for i in range(total_pages):
                        page = doc[i]
                        text = page.get_text()
                        if text:
                            clean_text = preprocess_text(text)
                            if clean_text and len(clean_text) > 10:
                                docs.append(
                                    Document(
                                        page_content=clean_text,
                                        metadata={
                                            "source": file_name,
                                            "page": i + 1,
                                            "total_pages": total_pages,
                                        },
                                    )
                                )
            else:
                # [최적화] 워커 수를 논리 코어 수에 맞추고 배치를 크게 설정하여 오버헤드 감소
                cpu_count = os.cpu_count() or 4
                max_workers = min(cpu_count, 16)

                # 너무 잦은 쓰레드 생성 방지를 위해 배치 크기 상향
                batch_size = max(4, total_pages // (max_workers * 2))
                batches = [
                    list(range(i, min(i + batch_size, total_pages)))
                    for i in range(0, total_pages, batch_size)
                ]

                all_results = [None] * total_pages
                completed_pages = 0

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            _extract_pages_batch_worker,
                            file_bytes,
                            batch,
                            total_pages,
                            file_name,
                        )
                        for batch in batches
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        batch_results = future.result()
                        for page_idx, doc in batch_results:
                            all_results[page_idx] = doc

                        completed_pages += len(batch_results)
                        SessionManager.replace_last_status_log(
                            f"텍스트 추출 중 ({min(completed_pages, total_pages)}/{total_pages}p)"
                        )
                        if on_progress:
                            import contextlib
                            with contextlib.suppress(Exception):
                                on_progress()

                docs = [r for r in all_results if r is not None]

            if not docs:
                raise EmptyPDFError(
                    filename=file_name,
                    details={"reason": "텍스트를 추출할 수 없습니다."}
                )

            SessionManager.replace_last_status_log(f"추출 완료 ({len(docs)}p)")
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            logger.info(f"[RAG] [LOAD] PDF 텍스트 추출 완료 | 파일: {file_name} | 페이지: {len(docs)}")
            return docs
        except (PDFProcessingError, EmptyPDFError):
            raise
        except Exception as e:
            logger.error(f"[RAG] [LOAD] PDF 로드 중 예상치 못한 오류 | {e}")
            raise PDFProcessingError(
                filename=file_name, details={"reason": str(e)}
            ) from e


def _split_documents(
    docs: list[Document],
    embedder: HuggingFaceEmbeddings | None = None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    """
    SessionManager.add_status_log("문장 분할 중...")
    with monitor.track_operation(
        OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
    ) as op:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)
        split_docs = []
        vectors = None

        if use_semantic and embedder:
            # [최적화] 디바이스 및 VRAM 상황에 따라 배치 크기 동적 할당
            import torch
            if getattr(embedder, "model_kwargs", {}).get("device") == "cuda":
                # GPU 환경: 가용 메모리에 따라 32~128 사이에서 결정
                try:
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
                    batch_size = 128 if total_mem > 10 else (64 if total_mem > 4 else 32)
                except Exception:
                    batch_size = 32
            else:
                # CPU 환경: 코어 수에 맞춰 4~16 사이에서 결정
                batch_size = min(max(4, os.cpu_count() or 4), 16)

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

            split_docs, vectors = semantic_chunker.split_documents(docs)
            logger.info(
                f"[RAG] [CHUNKING] 의미론적 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
            )
        else:
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            )
            split_docs = chunker.split_documents(docs)
            logger.info(f"[RAG] [CHUNKING] 기본 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}")

        # 청크 인덱스 메타데이터 추가
        for i, doc in enumerate(split_docs):
            doc.metadata = doc.metadata.copy()
            doc.metadata["chunk_index"] = i

        op.tokens = sum(len(doc.page_content.split()) for doc in split_docs)
        return split_docs, vectors


def _serialize_docs(docs: DocumentList) -> DocumentDictList:
    return [
        doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() for doc in docs
    ]


def _deserialize_docs(docs_as_dicts: DocumentDictList) -> DocumentList:
    from langchain_core.documents import Document

    return [Document(**d) for d in docs_as_dicts]


@functools.lru_cache(maxsize=1)
def _compute_config_hash() -> str:
    """설정 변경 감지용 해시 생성 (보안용 아님)"""
    config_dict = {
        "version": "2.1",  # 🚀 전처리 로직 변경으로 인한 캐시 무효화 강제
        "semantic_chunker": SEMANTIC_CHUNKER_CONFIG,
        "text_splitter": TEXT_SPLITTER_CONFIG,
        "retriever": RETRIEVER_CONFIG,
    }
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    # usedforsecurity=False: 이 해시는 보안 목적이 아닌 설정 변경 감지용임을 명시
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class VectorStoreCache:
    """벡터 저장소 및 리트리버 캐시 관리자"""

    # [추가] 프로세스 내 스레드 간 경쟁 방지를 위한 공유 락
    _global_write_lock = threading.Lock()

    def __init__(self, file_path: str, embedding_model_name: str, file_hash: str | None = None):
        self._file_hash = file_hash or _compute_file_hash(file_path)
        (
            self.cache_dir,
            self.doc_splits_path,
            self.faiss_index_path,
            self.bm25_retriever_path,
        ) = self._get_cache_paths(self._file_hash, embedding_model_name)

        # 캐시 보안 관리자 초기화
        self.security_manager = CacheSecurityManager(
            security_level=CACHE_SECURITY_LEVEL,
            hmac_secret=CACHE_HMAC_SECRET,
            trusted_paths=CACHE_TRUSTED_PATHS,
            check_permissions=CACHE_CHECK_PERMISSIONS,
        )

    def _get_cache_paths(
        self, file_hash: str, embedding_model_name: str
    ) -> tuple[str, str, str, str]:
        # 파일 경로에 안전하지 않은 문자 제거
        model_name_slug = embedding_model_name.replace("/", "_").replace("\\", "_")
        config_hash = _compute_config_hash()

        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}_{config_hash}"
        )

        return (
            cache_dir,
            os.path.join(cache_dir, "doc_splits.json"),  # [.pkl -> .json]
            os.path.join(cache_dir, "faiss_index"),
            os.path.join(cache_dir, "bm25_docs.json"),   # [.pkl -> .json]
        )

    def _purge_cache(self, reason: str):
        """보안 위협이나 손상이 감지된 캐시 디렉토리를 완전히 삭제합니다."""
        if os.path.exists(self.cache_dir):
            try:
                import shutil

                shutil.rmtree(self.cache_dir)
                logger.critical(
                    f"[Security] 캐시 강제 삭제됨 ({reason}): {self.cache_dir}"
                )
            except Exception as e:
                logger.error(f"캐시 삭제 실패: {e}")

    def load(
        self,
        embedder: HuggingFaceEmbeddings,
    ) -> tuple[list[Document] | None, FAISS | None, BM25Retriever | None]:
        """
        캐시된 RAG 컴포넌트를 로드합니다.

        보안 정책:
        - 신뢰 경로 위반(CacheTrustError) 또는 무결성 위반(CacheIntegrityError) 시
          보안 수준에 상관없이 캐시를 즉시 삭제하고 로드를 중단합니다.
        """
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
            from langchain_community.vectorstores import FAISS

            # --- 1. 보안 및 신뢰 검증 ---
            # 모든 주요 캐시 컴포넌트에 대해 루프를 돌며 검증 수행
            paths_to_verify = [
                (self.doc_splits_path, "문서 데이터"),
                (self.faiss_index_path, "FAISS 인덱스"),
                (self.bm25_retriever_path, "BM25 리트리버"),
            ]

            for path, desc in paths_to_verify:
                try:
                    # 신뢰 경로 검증 (모든 파일/폴더)
                    self.security_manager.verify_cache_trust(path)

                    # 무결성 검증 (파일인 경우에만 SHA256 체크)
                    if os.path.isfile(path):
                        # [보안 강화] 고보안 레벨에서는 HMAC 검증이 실패하거나 비밀키가 없으면 로드를 거부함
                        if CACHE_SECURITY_LEVEL == "high" and not CACHE_HMAC_SECRET:
                            raise CacheIntegrityError(
                                f"보안 레벨 'high'에서는 HMAC 비밀키가 필수입니다: {path}"
                            )
                        self.security_manager.verify_cache_integrity(path)
                    elif os.path.isdir(path):
                        # FAISS 인덱스 디렉토리 내의 핵심 파일 검증
                        index_file = os.path.join(path, "index.faiss")
                        if os.path.exists(index_file):
                            self.security_manager.verify_cache_integrity(index_file)

                except (CacheTrustError, CacheIntegrityError) as e:
                    self._purge_cache(
                        reason=f"Security Violation in {desc}: {type(e).__name__}"
                    )
                    return None, None, None
                except CachePermissionError as e:
                    if CACHE_SECURITY_LEVEL == "high":
                        self._purge_cache(reason=f"Permission Violation in {desc}")
                        return None, None, None
                    logger.warning(f"캐시 권한 경고 ({desc}): {e}")

            # --- 2. 데이터 로드 (모든 검증 통과 후) ---
            # 1. 문서 로드 (Pickle 대신 JSON 사용)
            with open(self.doc_splits_path, encoding="utf-8") as f:
                doc_dicts = json.load(f)
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 로드
            # 이미 위에서 무결성/신뢰 검증을 마쳤으므로 안전하게 로드
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            # 3. BM25 로드 (Pickle을 통한 상태 복원)
            # JSON 대신 .pkl 파일이 있으면 이를 우선 사용하여 인덱싱 시간 단축
            bm25_retriever_path_pkl = self.bm25_retriever_path.replace(".json", ".pkl")
            if os.path.exists(bm25_retriever_path_pkl):
                with open(bm25_retriever_path_pkl, "rb") as f:
                    bm25_retriever = pickle.load(f)  # nosec
            else:
                # 폴백: JSON으로부터 재구축
                with open(self.bm25_retriever_path, encoding="utf-8") as f:
                    bm25_doc_dicts = json.load(f)
                bm25_docs = _deserialize_docs(bm25_doc_dicts)

                from langchain_community.retrievers import BM25Retriever
                # [최적화] 모듈 레벨에 정의된 bm25_tokenizer 사용
                bm25_retriever = BM25Retriever.from_documents(bm25_docs, preprocess_func=bm25_tokenizer)

            bm25_retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]

            logger.info(f"RAG 캐시 안전 로드 완료: '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever

        except Exception as e:
            logger.warning(f"캐시 로드 중 예외 발생: {e}. 캐시를 폐기합니다.")
            self._purge_cache(reason="Load Error / Corruption")
            return None, None, None

    def save(
        self,
        doc_splits: DocumentList,
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
    ) -> None:
        """
        RAG 컴포넌트를 캐시에 안전하게 저장합니다. (원자적 저장 방식 적용)
        """
        import shutil
        import uuid

        # 1. 이미 다른 프로세스/스레드에 의해 완성되었는지 확인
        if os.path.exists(self.cache_dir):
            logger.info(f"[Cache] 캐시가 이미 존재함: {self.cache_dir}")
            return

        # 2. 임시 스테이징 디렉터리 생성 (고유 이름 부여)
        staging_dir = f"{self.cache_dir}.tmp.{uuid.uuid4().hex[:8]}"

        # 스테이징용 경로들 재구성
        stg_doc_splits_path = os.path.join(staging_dir, "doc_splits.json")
        stg_faiss_index_path = os.path.join(staging_dir, "faiss_index")
        stg_bm25_retriever_path = os.path.join(staging_dir, "bm25_docs.json")

        try:
            os.makedirs(staging_dir, exist_ok=True)

            # --- A. 스테이징 디렉터리에 데이터 작성 ---

            # 1. 문서 저장
            serialized_splits = _serialize_docs(doc_splits)
            with open(stg_doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(serialized_splits, f, ensure_ascii=False)

            doc_meta = self.security_manager.create_metadata_for_file(
                stg_doc_splits_path, description="Document splits cache (JSON)"
            )
            self.security_manager.save_cache_metadata(stg_doc_splits_path + ".meta", doc_meta)

            # 2. FAISS 저장
            vector_store.save_local(stg_faiss_index_path)
            for filename in ["index.faiss", "index.pkl"]:
                file_p = os.path.join(stg_faiss_index_path, filename)
                if os.path.exists(file_p):
                    meta = self.security_manager.create_metadata_for_file(
                        file_p, description=f"FAISS index part: {filename}"
                    )
                    self.security_manager.save_cache_metadata(file_p + ".meta", meta)

            # 3. BM25 저장
            # JSON은 호환성 및 무결성 체크용으로 유지
            bm25_docs = getattr(bm25_retriever, "docs", doc_splits)
            serialized_bm25 = _serialize_docs(bm25_docs)
            with open(stg_bm25_retriever_path, "w", encoding="utf-8") as f:
                json.dump(serialized_bm25, f, ensure_ascii=False)

            # [추가] 인덱스 상태가 포함된 Pickle 저장 (초고속 로딩용)
            stg_bm25_retriever_path_pkl = stg_bm25_retriever_path.replace(".json", ".pkl")
            with open(stg_bm25_retriever_path_pkl, "wb") as f:
                pickle.dump(bm25_retriever, f)

            # 메타데이터 생성 (Pickle 파일 포함)
            for p in [stg_bm25_retriever_path, stg_bm25_retriever_path_pkl]:
                bm25_meta = self.security_manager.create_metadata_for_file(
                    p, description=f"BM25 retriever data ({os.path.basename(p)})"
                )
                self.security_manager.save_cache_metadata(p + ".meta", bm25_meta)

            # --- B. 원자적 교체 (Atomic Rename) ---
            with self._global_write_lock:
                if not os.path.exists(self.cache_dir):
                    try:
                        os.rename(staging_dir, self.cache_dir)
                        logger.info(f"RAG 캐시 원자적 저장 및 검증 완료: '{self.cache_dir}'")
                    except Exception as e:
                        logger.error(f"캐시 최종 교체 실패: {e}")
                        raise
                else:
                    # 락 획득 대기 중에 다른 스레드가 먼저 저장한 경우
                    logger.info("[Cache] 다른 세션에 의해 캐시가 이미 생성됨. 스테이징 삭제.")
                    shutil.rmtree(staging_dir)

        except Exception as e:
            logger.error(f"캐시 저장 중 예외 발생: {e}")
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
            raise


@log_operation("FAISS 벡터 저장소 생성")
def _create_vector_store(
    docs: list[Document],
    embedder: HuggingFaceEmbeddings,
    vectors: Any = None,
) -> FAISS:
    """
    FAISS 벡터 저장소를 생성합니다.
    [최적화] 데이터 규모에 따라 Flat 또는 HNSW 인덱스를 자동 선택하고 코사인 유사도를 적용합니다.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    # 1. 임베딩 데이터 준비 및 정규화 (코사인 유사도용)
    if vectors is None:
        texts = [d.page_content for d in docs]
        vectors_list = embedder.embed_documents(texts)
        vectors = np.array(vectors_list).astype("float32")
        # [최적화] 외부에서 계산된 벡터가 없을 때만 정규화 수행
        import faiss
        faiss.normalize_L2(vectors)
    else:
        # [최적화] 이미 정규화된 numpy array로 변환 (추가 정규화 생략)
        vectors = np.array(vectors).astype("float32")

    normalized_vectors: np.ndarray = vectors

    text_embeddings = list(
        zip([d.page_content for d in docs], normalized_vectors, strict=False)
    )
    metadatas = [d.metadata for d in docs]

    # 2. 데이터 규모에 따른 인덱스 전략 결정
    chunk_count = len(docs)

    # LangChain FAISS에서는 distance_strategy를 통해 내부 메트릭 결정
    # METRIC_INNER_PRODUCT + Normalized Vectors = Cosine Similarity
    distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT

    if chunk_count >= 1000:
        # HNSW 적용 (대규모 고속 검색)
        logger.info(f"[FAISS] HNSW 인덱스 구축 시작 (Chunks: {chunk_count})")

        import uuid

        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_core.documents import Document

        d = normalized_vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128

        index.add(normalized_vectors)

        # [최적화] 루프 내 add 대신 딕셔너리 컴프리헨션으로 벌크 생성
        doc_ids = [str(uuid.uuid4()) for _ in range(chunk_count)]
        new_docs = {
            doc_id: Document(page_content=content, metadata=metadata)
            for doc_id, content, metadata in zip(
                doc_ids, [d.page_content for d in docs], metadatas, strict=False
            )
        }
        docstore = InMemoryDocstore(new_docs)
        index_to_docstore_id = dict(enumerate(doc_ids))

        vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=distance_strategy
        )
        logger.info(f"[RAG] [INDEX] FAISS HNSW 벌크 인덱스 구축 완료 | 청크: {chunk_count}")

    else:
        # 소규모는 정확한 검색(Flat) 수행
        logger.debug(f"[RAG] [INDEX] FAISS Flat 인덱스 사용 | 청크: {chunk_count}")

        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embedder,
            metadatas=metadatas,
            distance_strategy=distance_strategy
        )

    return vector_store


def _create_bm25_retriever(docs: list[Document]) -> BM25Retriever:
    from langchain_community.retrievers import BM25Retriever

    # [최적화] BM25 생성 시 형태소 분석기 대신 모듈 레벨에 정의된 bm25_tokenizer 사용
    retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenizer)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever


def _create_ensemble_retriever(
    vector_store: FAISS,
    bm25_retriever: BM25Retriever,
) -> EnsembleRetriever:
    from langchain.retrievers import EnsembleRetriever

    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=RETRIEVER_CONFIG["ensemble_weights"],
    )


@log_operation("검색 컴포넌트 로드/생성")
def _load_and_build_retrieval_components(
    file_path: str,
    file_name: str,
    _embedder: HuggingFaceEmbeddings,
    embedding_model_name: str,
    _on_progress=None,
    _file_hash: str | None = None
) -> tuple[DocumentList, FAISS, BM25Retriever, bool]:
    # 1. [최적화] 파일 통합 로드 및 해시 계산
    # 해시가 주어지지 않았다면 파일을 미리 읽어 해시 계산과 로딩에 공유
    file_bytes = None
    if _file_hash is None:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            _file_hash = _compute_file_hash(file_path, data=file_bytes)
        except Exception as e:
            logger.error(f"파일 통합 로드 실패: {e}")
            # 실패 시 기존 방식(파일 경로 기반)으로 폴백하기 위해 None 유지

    cache = VectorStoreCache(file_path, embedding_model_name, file_hash=_file_hash)
    doc_splits, vector_store, bm25_retriever = cache.load(_embedder)

    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        import numpy as np
        import torch

        # [최적화] 이미 읽어둔 바이트가 있다면 활용
        if file_bytes is not None:
            # _load_pdf_docs를 직접 호출하는 대신 바이트 기반 로직 수행 가능하지만,
            # 구조 유지를 위해 _load_pdf_docs 내부에 바이트 지원 로직 추가 필요
            # 여기서는 일관성을 위해 _load_pdf_docs 호출 (이미 내부에서 최적화됨)
            docs = _load_pdf_docs(file_path, file_name, on_progress=_on_progress)
        else:
            docs = _load_pdf_docs(file_path, file_name, on_progress=_on_progress)
        # 빈 문서 처리
        if not docs:
            raise EmptyPDFError(
                filename=file_name,
                details={"reason": "PDF에서 텍스트를 추출할 수 없습니다."},
            )

        if _on_progress:
            _on_progress()

        # [최적화 1] 문서 크기에 따른 바이패스 전략 (2000자 미만은 고속 처리)
        total_text_len = sum(len(d.page_content) for d in docs)
        is_small_doc = total_text_len < 2000

        # [최적화 2] 1차 분할 및 임베딩 생성 (단일 패스)
        doc_splits, precomputed_vectors = _split_documents(docs, _embedder)
        if _on_progress:
            _on_progress()

        if not doc_splits:
            raise InsufficientChunksError(chunk_count=0, min_required=1)

        # [최적화 3] 벡터 재사용을 통한 인덱스 최적화
        optimized_vectors: Any = precomputed_vectors
        q_meta = None
        optimizer = None

        if not is_small_doc:
            try:
                SessionManager.add_status_log("인덱스 최적화 중")
                if _on_progress:
                    _on_progress()

                # 벡터가 없으면(기본 분할기 사용 시) 한 번만 생성
                if optimized_vectors is None:
                    texts = [d.page_content for d in doc_splits]
                    vectors = _embedder.embed_documents(texts)
                    optimized_vectors = [np.array(v) for v in vectors]

                optimizer = get_index_optimizer()
                doc_splits, optimized_vectors, q_meta, stats = optimizer.optimize_index(
                    doc_splits, optimized_vectors
                )

                # [수정] 양자화된 벡터를 원래의 스케일로 복원하여 검색 정확도 보장
                if optimizer and q_meta and q_meta.get("method") != "none":
                    optimized_vectors = optimizer.quantizer.dequantize_vectors(
                        optimized_vectors, q_meta
                    )

                logger.info(f"인덱스 최적화 완료: 중복 {stats.pruned_documents}개 제거")
                SessionManager.replace_last_status_log(
                    f"중복 내용 {stats.pruned_documents}개 정리"
                )
                if _on_progress:
                    _on_progress()
            except Exception as e:
                logger.warning(f"인덱스 최적화 단계 건너뜀 (경미한 오류): {e}")
                # 오류 발생 시에도 doc_splits와 optimized_vectors는 유지됨

        # [최적화 4] 계산된 벡터를 FAISS에 직접 주입 (GPU 추가 호출 0회)
        vector_store = _create_vector_store(
            doc_splits, _embedder, vectors=optimized_vectors
        )
        bm25_retriever = _create_bm25_retriever(doc_splits or [])

        # 캐시 저장
        cache.save(doc_splits, vector_store, bm25_retriever)

        # [최적화 5] GPU 자원 즉시 반환 (Ollama와의 VRAM 경합 방지)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("[System] [Memory] CUDA 캐시 비우기 완료")
            except Exception:
                pass

        SessionManager.add_status_log("신규 인덱싱 완료")

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str,
    file_path: str,
    embedder: HuggingFaceEmbeddings,
    on_progress=None,
) -> tuple[str, bool]:
    """
    RAG 파이프라인을 구축하고 세션에 저장합니다.
    """
    # [최적화] 파일 해시는 여기서 한 번만 계산하여 하위 함수로 전달
    file_hash = _compute_file_hash(file_path)

    # [최적화] embedder 객체는 해싱에서 제외하고, 모델명과 파일 해시를 명시적 키로 전달
    doc_splits, vector_store, bm25_retriever, cache_used = (
        _load_and_build_retrieval_components(
            file_path,
            uploaded_file_name,
            _embedder=embedder,
            embedding_model_name=embedder.model_name,
            _on_progress=on_progress,
            _file_hash=file_hash
        )
    )

    if cache_used:
        SessionManager.add_status_log("캐시 데이터 로드")
        if on_progress:
            on_progress()

    # [최적화] 병렬 검색을 위해 개별 리트리버 생성 및 세션 저장
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )

    SessionManager.set("faiss_retriever", faiss_retriever)
    SessionManager.set("bm25_retriever", bm25_retriever)

    # 기존 호환성 유지 (EnsembleRetriever도 생성)
    from langchain.retrievers import EnsembleRetriever

    final_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=RETRIEVER_CONFIG["ensemble_weights"],
    )
    rag_engine = build_graph(retriever=final_retriever)

    SessionManager.set("vector_store", vector_store)
    SessionManager.set("rag_engine", rag_engine)
    SessionManager.set("pdf_processed", True)
    SessionManager.add_status_log("질문 가능")

    if on_progress:
        on_progress()

    logger.info(
        f"RAG 파이프라인 구축 완료: '{uploaded_file_name}' (캐시 사용: {cache_used})"
    )

    if cache_used:
        return f"'{uploaded_file_name}' 문서 캐시 데이터 로드 완료", True
    return (
        f"'{uploaded_file_name}' 신규 문서 인덱싱 완료"
    ), False


@log_operation("파이프라인 LLM 업데이트")
def update_llm_in_pipeline(llm: T | None) -> None:
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
