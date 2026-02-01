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
    """페이지 범위를 배치로 처리하는 워커 함수 (메모리 스트림 사용으로 파일 잠금 방지)"""
    results = []
    try:
        import fitz  # PyMuPDF
        from langchain_core.documents import Document

        # [개선] 파일 경로 대신 메모리 버퍼로부터 문서를 열어 공유 위반 방지
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
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
    PDF 파일을 배치 기반 병렬 로드로 변환하여 최적화합니다.
    """
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            import fitz  # PyMuPDF

            SessionManager.add_status_log("문서 분석 준비 중")
            if on_progress:
                on_progress()

            try:
                doc_file = fitz.open(file_path)
                total_pages = len(doc_file)
                doc_file.close()
            except Exception as e:
                raise PDFProcessingError(
                    filename=file_name,
                    details={"reason": f"파일을 열 수 없음: {str(e)}"},
                ) from e

            if total_pages == 0:
                raise EmptyPDFError(
                    filename=file_name, details={"reason": "PDF 페이지가 없습니다."}
                )

            # [개선] 파일을 한 번만 읽어 메모리에 버퍼링 (파일 잠금 원천 방지 및 I/O 최적화)
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # [최적화] 페이지 수에 따라 동적으로 병렬화 결정 (임계값 5p로 하향)
            if total_pages <= 5:
                docs = []
                # [개선] 단일 스레드에서도 메모리 스트림 사용
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
                        if (i + 1) % 5 == 0 or i == total_pages - 1:
                            SessionManager.replace_last_status_log(
                                f"텍스트 추출 중 ({i + 1}/{total_pages}p)"
                            )
                            if on_progress:
                                on_progress()
            else:
                SessionManager.replace_last_status_log(
                    f"병렬 배치 추출 시작 (총 {total_pages}p)"
                )
                if on_progress:
                    on_progress()

                # [최적화] 워커 수를 CPU 코어 수에 맞춰 더 적극적으로 할당
                max_workers = min(os.cpu_count() or 4, 12)
                # 페이지를 더 작게 쪼개어 부하 분산
                batch_size = max(1, total_pages // (max_workers * 2))
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
                            file_bytes,  # [수정] 파일 경로 대신 메모리 데이터 전달
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
                        # [수정] SessionManager 업데이트는 스레드 세이프하므로 유지하되,
                        # UI 갱신 콜백(on_progress)은 메인 루프인 이곳에서 호출하여 안전성 확보
                        SessionManager.replace_last_status_log(
                            f"텍스트 추출 중 ({min(completed_pages, total_pages)}/{total_pages}p)"
                        )
                        if on_progress:
                            try:
                                on_progress()
                            except Exception as e:
                                logger.debug(
                                    f"UI 업데이트 건너뜀 (Context missing): {e}"
                                )

                docs = [r for r in all_results if r is not None]

            if not docs:
                raise EmptyPDFError(
                    filename=file_name,
                    details={
                        "reason": "추출된 텍스트가 없습니다. 이미지 기반 PDF일 수 있습니다."
                    },
                )

            SessionManager.replace_last_status_log(f"추출 완료 (구간 {len(docs)}개)")
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs
        except (PDFProcessingError, EmptyPDFError):
            raise
        except Exception as e:
            logger.error(f"PDF 로드 중 예상치 못한 오류: {e}")
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
            # [최적화] 무거운 자동 배치 최적화 대신 고정값 사용
            batch_size = (
                32
                if getattr(embedder, "model_kwargs", {}).get("device") == "cuda"
                else 4
            )

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
                f"의미론적 분할 완료: {len(docs)} 문서 -> {len(split_docs)} 청크 (벡터 재사용 준비 완료)"
            )
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
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_dicts = json.load(f)
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 로드
            # 이미 위에서 무결성/신뢰 검증을 마쳤으므로 안전하게 로드
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            # 3. BM25 로드 (Rebuild from JSON docs)
            with open(self.bm25_retriever_path, "r", encoding="utf-8") as f:
                bm25_doc_dicts = json.load(f)
            bm25_docs = _deserialize_docs(bm25_doc_dicts)
            
            from langchain_community.retrievers import BM25Retriever
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]

            logger.info(f"RAG 캐시 안전 로드 완료 (JSON): '{self.cache_dir}'")
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
        RAG 컴포넌트를 캐시에 저장합니다.
        저장 후 파일 존재 여부를 검증하여 무결성을 보장합니다.
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # 1. 문서 저장 (Pickle 대신 JSON 사용)
            try:
                serialized_splits = _serialize_docs(doc_splits)
                with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_splits, f, ensure_ascii=False)

                # 문서 캐시 메타데이터 생성
                doc_meta = self.security_manager.create_metadata_for_file(
                    self.doc_splits_path, description="Document splits cache (JSON)"
                )
                self.security_manager.save_cache_metadata(
                    self.doc_splits_path + ".meta", doc_meta
                )
            except Exception as e:
                logger.error(f"문서 데이터 저장 실패: {e}")
                raise OSError(f"Failed to save doc_splits: {e}") from e

            # 2. FAISS 저장
            try:
                vector_store.save_local(self.faiss_index_path)

                # [수정] FAISS 내부 핵심 파일들(index.faiss, index.pkl)에 대한 메타데이터 생성
                for filename in ["index.faiss", "index.pkl"]:
                    file_p = os.path.join(self.faiss_index_path, filename)
                    if os.path.exists(file_p):
                        meta = self.security_manager.create_metadata_for_file(
                            file_p, description=f"FAISS index part: {filename}"
                        )
                        self.security_manager.save_cache_metadata(
                            file_p + ".meta", meta
                        )
            except Exception as e:
                logger.error(f"FAISS 인덱스 저장 실패: {e}")
                raise OSError(f"Failed to save FAISS index: {e}") from e

            # 3. BM25 저장 (JSON 사용 - 내부 문서를 추출하여 저장)
            try:
                # BM25Retriever에서 문서 추출
                bm25_docs = getattr(bm25_retriever, "docs", doc_splits)
                serialized_bm25 = _serialize_docs(bm25_docs)
                
                with open(self.bm25_retriever_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_bm25, f, ensure_ascii=False)

                metadata = self.security_manager.create_metadata_for_file(
                    self.bm25_retriever_path, description="BM25 docs cache (JSON)"
                )

                if CACHE_SECURITY_LEVEL == "high" and CACHE_HMAC_SECRET:
                    with open(self.bm25_retriever_path, "rb") as f:
                        file_data = f.read()
                    metadata.integrity_hmac = (
                        self.security_manager.compute_integrity_hmac(file_data)
                    )

                self.security_manager.save_cache_metadata(
                    self.bm25_retriever_path + ".meta", metadata
                )
            except Exception as e:
                logger.error(f"BM25 리트리버 데이터 저장 실패: {e}")
                raise OSError(f"Failed to save BM25 docs: {e}") from e

            # --- 검증 단계 (Verification) ---
            required_paths = [
                self.doc_splits_path,
                os.path.join(self.faiss_index_path, "index.faiss"),
                os.path.join(self.faiss_index_path, "index.pkl"),
                self.bm25_retriever_path,
            ]

            missing = [p for p in required_paths if not os.path.exists(p)]
            if missing:
                error_msg = f"캐시 저장 검증 실패. 누락된 파일: {missing}"
                logger.error(error_msg)
                self._purge_cache(reason="Verification Failed After Save")
                raise OSError(error_msg)

            logger.info(f"RAG 캐시 저장 및 검증 완료: '{self.cache_dir}'")
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            self._purge_cache(reason=f"Exception during save: {type(e).__name__}")
            raise


@log_operation("FAISS 벡터 저장소 생성")
def _create_vector_store(
    docs: list[Document],
    embedder: HuggingFaceEmbeddings,
    vectors: list[np.ndarray] | None = None,
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
    else:
        # [수정] list[np.ndarray] 형태인 경우 2D numpy array로 변환
        if isinstance(vectors, list):
            vectors = np.array(vectors).astype("float32")

    # [최적화] FAISS의 C++ 최적화 정규화 함수 사용 (SIMD 가속)
    import faiss
    faiss.normalize_L2(vectors)
    normalized_vectors = vectors

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

        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import uuid

        d = normalized_vectors.shape[1]
        # HNSW 파라미터: m=32, ef_construction=128
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        
        # 벡터 추가
        index.add(normalized_vectors)
        
        # Docstore 및 ID 맵 생성
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        
        for i, (content, metadata) in enumerate(zip([d.page_content for d in docs], metadatas, strict=False)):
            doc_id = str(uuid.uuid4())
            doc = Document(page_content=content, metadata=metadata)
            docstore.add({doc_id: doc})
            index_to_docstore_id[i] = doc_id
            
        vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=distance_strategy
        )
        logger.info("[FAISS] HNSW 최적화 인덱스 직접 구축 완료")
    else:
        # 소규모는 정확한 검색(Flat) 수행
        logger.debug(f"[FAISS] Flat 인덱스 사용 (Chunks: {chunk_count})")
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embedder,
            metadatas=metadatas,
            distance_strategy=distance_strategy
        )

    return vector_store


def _create_bm25_retriever(docs: list[Document]) -> BM25Retriever:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs)
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
    cache = VectorStoreCache(file_path, embedding_model_name, file_hash=_file_hash)
    doc_splits, vector_store, bm25_retriever = cache.load(_embedder)

    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
        import numpy as np
        import torch

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
        optimized_vectors = precomputed_vectors
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
                if q_meta and q_meta.get("method") != "none":
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
