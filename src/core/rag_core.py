"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
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
    from langchain_classic.retrievers import EnsembleRetriever
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


import re

# --- 최적화된 토크나이저 ---
_RE_KOREAN_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+")


def bm25_tokenizer(text: str) -> list[str]:
    """
    [최적화] 한국어 검색 품질 향상을 위한 Hybrid 토크나이저.
    기본 정규식 추출 + 어미 제거 + Bi-gram 생성을 수행합니다.
    """
    if not text:
        return []

    # 1. 기본 토큰 추출
    tokens = _RE_KOREAN_TOKEN.findall(text.lower())
    if not tokens:
        return text.split()

    final_tokens = []
    # 자주 쓰이는 조사/어미 (간이 불용어 처리)
    particles = ("은", "는", "이", "가", "을", "를", "의", "에", "로", "서", "들")

    for token in tokens:
        final_tokens.append(token)

        # 한글인 경우 추가 처리
        if "가" <= token[0] <= "힣":
            # 2. 간단한 어미/조사 제거 (끝글자 체크)
            if len(token) > 2 and token.endswith(particles):
                stem = token[:-1]
                if len(stem) >= 2:
                    final_tokens.append(stem)

            # 3. Bi-gram 생성 (3글자 이상인 경우)
            # 복합명사 검색 재현율 향상 (예: 인공지능 -> 인공, 공지, 지능)
            if len(token) >= 3:
                for i in range(len(token) - 1):
                    final_tokens.append(token[i : i + 2])

    return final_tokens


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


import concurrent.futures

import fitz  # PyMuPDF
from langchain_core.documents import Document


def _extract_page_worker(
    file_path: str, page_num: int, total_pages: int, file_name: str
) -> Document | None:
    """개별 페이지에서 텍스트를 추출하는 워커 함수 (스레드 세이프)"""
    try:
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
    file_bytes: bytes | None,
    file_path: str,
    page_range: list[int],
    total_pages: int,
    file_name: str,
) -> list[tuple[int, Document]]:
    """페이지 범위를 배치로 처리하는 워커 함수 (하이브리드 로딩 지원)"""
    results = []
    doc = None
    try:
        # [최적화] 하이브리드 로딩 전략
        # 작은 파일 -> 메모리뷰 (속도), 큰 파일 -> 파일 경로 (메모리 절약)
        if file_bytes is not None:
            mv = memoryview(file_bytes)
            doc = fitz.open(stream=mv, filetype="pdf")
        else:
            doc = fitz.open(file_path)

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
                                Document(page_content=clean_text, metadata=metadata),
                            )
                        )
            except Exception as e:
                logger.warning(f"페이지 {page_num + 1} 추출 실패: {e}")
    except Exception as e:
        logger.error(f"배치 처리 중 문서 오픈 실패: {e}")
    finally:
        if doc:
            doc.close()
    return results


def _load_pdf_docs(
    file_path: str,
    file_name: str,
    on_progress: Callable[[], None] | None = None,
    file_bytes: bytes | None = None,
    session_id: str | None = None,
) -> list[Document]:
    """
    PDF 파일을 메모리에 버퍼링한 후 병렬 배치 처리를 통해 최고 속도로 추출합니다.
    [최적화] 50MB 이상 파일은 스트리밍 방식으로 전환하여 메모리 폭증 방지.
    """
    # 메모리 로딩 제한 설정 (50MB)
    MEMORY_LOAD_LIMIT = 50 * 1024 * 1024

    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log("문서 분석 준비 중", session_id=session_id)
            if on_progress:
                on_progress()

            file_size = os.path.getsize(file_path)
            use_memory_loading = file_size <= MEMORY_LOAD_LIMIT

            # 1. 파일 로딩 전략 결정
            # 이미 로드된 바이트가 없고, 파일이 작으면 메모리에 로드
            try:
                if file_bytes is None and use_memory_loading:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()

                # 큰 파일은 file_bytes를 None으로 유지하여 워커가 직접 읽게 함
                if not use_memory_loading:
                    file_bytes = None
                    logger.info(
                        f"[PDF] 대용량 파일 감지({file_size / 1024 / 1024:.1f}MB). 직접 접근 모드 사용."
                    )

                # 메타데이터 확인용 (경량 오픈)
                doc_chk = fitz.open(file_path)
                total_pages = len(doc_chk)
                doc_chk.close()

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
                # 작은 문서는 메인 스레드에서 처리
                doc = fitz.open(file_path)
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
                doc.close()
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
                            file_bytes,  # 큰 파일이면 None 전달
                            file_path,  # 파일 경로 전달
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
                            f"텍스트 추출 중 ({min(completed_pages, total_pages)}/{total_pages}p)",
                            session_id=session_id,
                        )
                        if on_progress:
                            import contextlib

                            with contextlib.suppress(Exception):
                                on_progress()

                docs = [r for r in all_results if r is not None]

            if not docs:
                raise EmptyPDFError(
                    filename=file_name,
                    details={"reason": "텍스트를 추출할 수 없습니다."},
                )

            SessionManager.replace_last_status_log(
                f"추출 완료 ({len(docs)}p)", session_id=session_id
            )
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            logger.info(
                f"[RAG] [LOAD] PDF 텍스트 추출 완료 | 파일: {file_name} | 페이지: {len(docs)}"
            )
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
    session_id: str | None = None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    """
    SessionManager.add_status_log("문장 분할 중...", session_id=session_id)
    with monitor.track_operation(
        OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
    ) as op:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)
        split_docs = []
        vectors = None

        if use_semantic and embedder:
            # [최적화] 디바이스 및 VRAM 상황에 따라 배치 크기 동적 할당
            import torch

            if getattr(embedder, "model_kwargs", {}).get("device") == "cuda":
                # GPU 환경: 가용 메모리에 따라 32~128 사이에서 결정
                try:
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )  # GB
                    batch_size = (
                        128 if total_mem > 10 else (64 if total_mem > 4 else 32)
                    )
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
            logger.info(
                f"[RAG] [CHUNKING] 기본 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
            )

            # [최적화] 기본 분할 시에도 벡터를 즉시 계산하여 중복 요청 방지
            if embedder and split_docs:
                texts = [d.page_content for d in split_docs]
                vectors_list = embedder.embed_documents(texts)
                vectors = [np.array(v) for v in vectors_list]

        # 청크 인덱스 및 본문 여부 메타데이터 추가
        noise_keywords = ["index", "references", "bibliography", "doi:", "isbn"]
        for i, doc in enumerate(split_docs):
            doc.metadata = doc.metadata.copy()
            doc.metadata["chunk_index"] = i

            # [추가] 본문 여부 판별 (사전 필터링용)
            content_lower = doc.page_content.lower()
            is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
            # DOI 링크가 너무 많거나 콤마/숫자가 색인처럼 많으면 노이즈로 간주
            if not is_noise and (
                content_lower.count("doi:") > 2 or content_lower.count(",") > 25
            ):
                is_noise = True

            doc.metadata["is_content"] = not is_noise

        op.tokens = sum(len(doc.page_content.split()) for doc in split_docs)
        return split_docs, vectors


def _serialize_docs(docs: DocumentList) -> DocumentDictList:
    """[최적화] Pydantic의 무거운 dict() 대신 직접 필요한 필드만 추출하여 성능 향상"""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
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

    def __init__(
        self, file_path: str, embedding_model_name: str, file_hash: str | None = None
    ):
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
            os.path.join(cache_dir, "bm25_docs.json"),  # [.pkl -> .json]
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
            # 1. 문서 로드 (orjson 사용으로 속도 향상)
            import orjson

            with open(self.doc_splits_path, "rb") as f:
                doc_dicts = orjson.loads(f.read())
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 로드
            # 이미 위에서 무결성/신뢰 검증을 마쳤으므로 안전하게 로드
            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,  # [Security] 위에서 무결성 검증 완료됨
            )

            # 3. BM25 로드 (보안을 위해 Pickle 대신 JSON으로부터 재구축)
            # [수정] RCE 위험이 있는 pickle.load() 제거
            with open(self.bm25_retriever_path, "rb") as f:
                bm25_doc_dicts = orjson.loads(f.read())
            bm25_docs = _deserialize_docs(bm25_doc_dicts)

            from langchain_community.retrievers import BM25Retriever

            # [최적화] 모듈 레벨에 정의된 bm25_tokenizer 사용
            bm25_retriever = BM25Retriever.from_documents(
                bm25_docs, preprocess_func=bm25_tokenizer
            )

            bm25_retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]

            logger.info(f"RAG 캐시 안전 로드 완료 (JSON 기반): '{self.cache_dir}'")
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
            import orjson

            # --- A. 스테이징 디렉터리에 데이터 작성 ---

            # 1. 문서 저장 (orjson 사용)
            serialized_splits = _serialize_docs(doc_splits)
            with open(stg_doc_splits_path, "wb") as f:
                f.write(orjson.dumps(serialized_splits))

            doc_meta = self.security_manager.create_metadata_for_file(
                stg_doc_splits_path, description="Document splits cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_doc_splits_path + ".meta", doc_meta
            )

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
            bm25_docs = getattr(bm25_retriever, "docs", doc_splits)
            serialized_bm25 = _serialize_docs(bm25_docs)
            with open(stg_bm25_retriever_path, "wb") as f:
                f.write(orjson.dumps(serialized_bm25))

            # [보안] RCE 위험이 있는 Pickle 저장 로직 제거됨

            # 메타데이터 생성 (JSON 파일에 대해)
            bm25_meta = self.security_manager.create_metadata_for_file(
                stg_bm25_retriever_path, description="BM25 retriever data (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_bm25_retriever_path + ".meta", bm25_meta
            )

            # --- B. 원자적 교체 (Atomic Rename) ---
            with self._global_write_lock:
                if not os.path.exists(self.cache_dir):
                    try:
                        os.rename(staging_dir, self.cache_dir)
                        logger.info(
                            f"RAG 캐시 원자적 저장 및 검증 완료: '{self.cache_dir}'"
                        )
                    except Exception as e:
                        logger.error(f"캐시 최종 교체 실패: {e}")
                        raise
                else:
                    # 락 획득 대기 중에 다른 스레드가 먼저 저장한 경우
                    logger.info(
                        "[Cache] 다른 세션에 의해 캐시가 이미 생성됨. 스테이징 삭제."
                    )
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
    FAISS 벡터 저장소를 최적화된 방식으로 생성합니다.
    [최적화] Index Factory를 통해 SQ8(양자화) + HNSW 구조를 적용하여 메모리 절감 및 속도 향상.
    """
    import uuid

    import faiss
    import numpy as np
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    # 1. 임베딩 데이터 준비 및 정규화
    if vectors is None:
        logger.warning(
            "[FAISS] 전달된 벡터가 없어 임베딩을 다시 수행합니다 (비효율적)."
        )
        texts = [d.page_content for d in docs]
        vectors_list = embedder.embed_documents(texts)
        vectors = np.array(vectors_list).astype("float32")
    else:
        vectors = np.array(vectors).astype("float32")

    faiss.normalize_L2(vectors)
    chunk_count = len(docs)
    d = vectors.shape[1]

    # 2. 최적화된 인덱스 팩토리 설정
    # - Flat: 정밀도가 중요한 소규모 데이터
    # - HNSW32,SQ8: 메모리 효율과 속도가 중요한 대규모 데이터
    index_type = "Flat" if chunk_count < 1000 else "HNSW32,SQ8"

    # Inner Product(IP)는 정규화된 벡터에서 코사인 유사도와 동일함
    index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)

    # 3. 인덱스 학습 (양자화를 사용하는 경우 학습 단계 필요)
    if "SQ" in index_type or "IVF" in index_type:
        logger.info(f"[FAISS] 인덱스 학습 시작 ({index_type})")
        index.train(vectors)

    # 데이터 추가
    index.add(vectors)

    # 4. HNSW 세부 튜닝 (검색 시 정밀도 향상)
    if "HNSW" in index_type:
        # efSearch: 검색 시 후보 탐색 범위 (값이 클수록 정확하지만 느려짐)
        faiss.downcast_index(index).hnsw.efSearch = 128

    # 5. LangChain FAISS 객체로 래핑
    doc_ids = [str(uuid.uuid4()) for _ in range(chunk_count)]
    new_docs = {
        doc_id: Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc_id, doc in zip(doc_ids, docs, strict=False)
    }
    docstore = InMemoryDocstore(new_docs)
    index_to_docstore_id = dict(enumerate(doc_ids))

    return FAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


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
    from langchain_classic.retrievers import EnsembleRetriever

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
    _file_hash: str | None = None,
    session_id: str | None = None,
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
        import torch

        # [최적화] 이미 읽어둔 바이트가 있다면 활용 (중복 I/O 제거)
        docs = _load_pdf_docs(
            file_path,
            file_name,
            on_progress=_on_progress,
            file_bytes=file_bytes,
            session_id=session_id,
        )

        # [추가] 문서 언어 감지 (첫 1000자 기준)
        if docs:
            sample_text = docs[0].page_content[:1000]
            # 한글 포함 여부로 간단히 판별 (확장 가능)
            has_korean = any("\uac00" <= char <= "\ud7a3" for char in sample_text)
            doc_lang = "Korean" if has_korean else "English"
            SessionManager.set("doc_language", doc_lang, session_id=session_id)
            logger.info(f"[RAG] [LANG] 문서 언어 감지됨: {doc_lang}")

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
        doc_splits, precomputed_vectors = _split_documents(
            docs, _embedder, session_id=session_id
        )
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
                SessionManager.add_status_log("인덱스 최적화 중", session_id=session_id)
                if _on_progress:
                    _on_progress()

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
                    f"중복 내용 {stats.pruned_documents}개 정리", session_id=session_id
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

        SessionManager.add_status_log("신규 인덱싱 완료", session_id=session_id)

    return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str,
    file_path: str,
    embedder: HuggingFaceEmbeddings,
    on_progress=None,
    session_id: str | None = None,
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
            _file_hash=file_hash,
            session_id=session_id,
        )
    )

    if cache_used:
        SessionManager.add_status_log("캐시 데이터 로드", session_id=session_id)
        if on_progress:
            on_progress()

    # [최적화] 병렬 검색을 위해 개별 리트리버 생성 및 세션 저장
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )

    SessionManager.set("faiss_retriever", faiss_retriever, session_id=session_id)
    SessionManager.set("bm25_retriever", bm25_retriever, session_id=session_id)

    # 기존 호환성 유지 (EnsembleRetriever도 생성)
    from langchain_classic.retrievers import EnsembleRetriever

    final_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=RETRIEVER_CONFIG["ensemble_weights"],
    )
    rag_engine = build_graph(retriever=final_retriever)

    SessionManager.set("vector_store", vector_store, session_id=session_id)
    SessionManager.set("rag_engine", rag_engine, session_id=session_id)
    SessionManager.set("pdf_processed", True, session_id=session_id)
    SessionManager.add_status_log("질문 가능", session_id=session_id)

    if on_progress:
        on_progress()

    logger.info(
        f"RAG 파이프라인 구축 완료: '{uploaded_file_name}' (캐시 사용: {cache_used}, Session: {session_id})"
    )

    if cache_used:
        return f"'{uploaded_file_name}' 문서 캐시 데이터 로드 완료", True
    return (f"'{uploaded_file_name}' 신규 문서 인덱싱 완료"), False


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
