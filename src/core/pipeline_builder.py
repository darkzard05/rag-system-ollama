"""
RAG 파이프라인 구축·캐싱·쿼리 설정 로직.
RAGSystem에서 파이프라인 구축 및 실행 설정 책임을 분리하여 캡슐화합니다.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings

from cache.vector_cache import VectorStoreCache
from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ENABLE_VECTOR_CACHE,
    RETRIEVER_CONFIG,
)
from common.exceptions import EmptyPDFError, InsufficientChunksError, VectorStoreError
from core.chunking import split_documents
from core.document_processor import compute_file_hash, load_pdf_docs
from core.graph_builder import build_graph
from core.resource_manager import get_resource_manager
from core.retriever_factory import create_bm25_retriever, create_vector_store
from core.session import SessionManager

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    RAG 파이프라인 구축 전담 서비스.
    문서 로딩, 청크 분할, 벡터 스토어/리트리버 생성, 캐싱을 담당합니다.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    async def build(
        self,
        file_path: str,
        file_name: str,
        embedder: Embeddings,
        on_progress: Callable[[int], Any] | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        """문서를 로드하고 RAG 파이프라인을 구축합니다.

        Args:
            file_path: PDF 파일 경로.
            file_name: 원본 파일 이름.
            embedder: 임베딩 모델 인스턴스.
            on_progress: 진행률 콜백 (0~100).
            check_cancelled: True 반환 시 asyncio.CancelledError 발생으로 중단.

        Returns:
            (메시지, 캐시 사용 여부) 튜플.
        """

        def _check() -> None:
            if check_cancelled is not None and check_cancelled():
                raise asyncio.CancelledError(
                    "파이프라인 구축이 사용자에 의해 취소되었습니다"
                )

        start_time = time.time()
        logger.info(f"[RAG] [INDEX] 파이프라인 구축 시작: {file_name}")

        file_hash = compute_file_hash(file_path)
        SessionManager.set("file_hash", file_hash, session_id=self.session_id)

        emb_model_name = getattr(
            embedder, "model", getattr(embedder, "model_name", "unknown")
        )
        cache = VectorStoreCache(file_path, emb_model_name, file_hash=file_hash)

        # 1. 캐시 시도
        if ENABLE_VECTOR_CACHE:
            cache_data = cache.load(embedder)
            if cache_data and all(x is not None for x in cache_data):
                doc_splits, vector_store, bm25_retriever = cache_data
                if doc_splits is not None:
                    logger.info(
                        f"[RAG] [INDEX] 벡터 캐시 히트: {len(doc_splits)}개 청크 로드됨"
                    )

                    SessionManager.add_status_log(
                        "기존 분석 데이터 발견 (캐시 활용)", session_id=self.session_id
                    )
                    await self._register_and_finalize(
                        file_hash, vector_store, bm25_retriever, on_progress
                    )
                    return f"'{file_name}' 캐시 데이터 로드 완료", True

        # 2. 신규 문서 로드
        SessionManager.add_status_log(
            f"'{file_name}' 텍스트 추출 중...", session_id=self.session_id
        )
        documents = await load_pdf_docs(
            file_path,
            file_name,
            session_id=self.session_id,
            file_hash=file_hash,
        )
        if not documents:
            raise EmptyPDFError()

        # 3. 청크 분할 (Async)
        SessionManager.add_status_log(
            "문맥 최적화 및 청크 분할 중...", session_id=self.session_id
        )
        doc_splits, vectors = await split_documents(
            documents, embedder=embedder, session_id=self.session_id
        )
        if not doc_splits:
            raise InsufficientChunksError()

        _check()

        # 4–5. 벡터 스토어 및 BM25 리트리버 병렬 생성
        SessionManager.add_status_log(
            "지식 베이스(Vector Index) 생성 중...", session_id=self.session_id
        )
        vector_store_future = asyncio.to_thread(
            create_vector_store, doc_splits, embedder, vectors=vectors
        )
        bm25_future = asyncio.to_thread(create_bm25_retriever, doc_splits)
        vector_store, bm25_retriever = await asyncio.gather(
            vector_store_future, bm25_future
        )

        _check()

        # 6. 등록 및 최종화 (캐시 저장 전에 수행 — 실패 시 캐시 미저장)
        await self._register_and_finalize(
            file_hash, vector_store, bm25_retriever, on_progress
        )

        # 7. 캐시 저장 (엔진 등록 성공 후에만 저장)
        if ENABLE_VECTOR_CACHE:
            cache.save(doc_splits, vector_store, bm25_retriever)

        duration = time.time() - start_time
        logger.info(
            f"[RAG] [INDEX] 파이프라인 구축 완료: {file_name} ({duration:.2f}s)"
        )
        return f"'{file_name}' 분석 및 신규 인덱싱 완료", False

    async def _register_and_finalize(
        self,
        file_hash: str,
        vector_store: Any,
        bm25_retriever: Any,
        on_progress: Callable[[int], Any] | None = None,
    ) -> None:
        """생성된 리소스를 전역 풀에 등록하고 세션을 초기화합니다."""
        _, workflow = await asyncio.gather(
            get_resource_manager().register_retrievers(
                file_hash, vector_store, bm25_retriever
            ),
            build_graph(),
        )
        SessionManager.set("rag_engine", workflow, session_id=self.session_id)

        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0
        SessionManager.set(
            "rag_engine_loop_id", current_loop_id, session_id=self.session_id
        )

        if on_progress:
            on_progress(100)


async def prepare_query_config(
    session_id: str,
    model_name: str | None = None,
    build_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """LLM·임베더·리트리버 설정을 포함한 LangGraph 실행 Config를 준비합니다."""
    from core.model_loader import ModelManager

    target_model = model_name or DEFAULT_OLLAMA_MODEL
    selected_embedding = (
        SessionManager.get("last_selected_embedding_model", session_id=session_id)
        or DEFAULT_EMBEDDING_MODEL
    )

    llm = SessionManager.get("llm", session_id=session_id)
    embedder = SessionManager.get("embedder", session_id=session_id)

    if not llm or not embedder:
        llm, embedder = await asyncio.gather(
            ModelManager.get_llm(target_model),
            ModelManager.get_embedder(selected_embedding),
        )
        SessionManager.set("llm", llm, session_id=session_id)
        SessionManager.set("embedder", embedder, session_id=session_id)

    file_hash = SessionManager.get("file_hash", session_id=session_id)
    if not file_hash:
        raise VectorStoreError(details={"reason": "파일 해시 없음"})

    pdf_file_path = SessionManager.get("pdf_file_path", session_id=session_id)
    if pdf_file_path:
        coordinator = get_resource_manager()
        vector_store, bm25_shared = await coordinator.get_or_build(
            coordinator.retrievers,
            file_hash,
            build_fn=build_fn,
            file_path=pdf_file_path,
            file_name=SessionManager.get(
                "last_uploaded_file_name", session_id=session_id
            ),
            embedder=embedder,
        )
    else:
        result = get_resource_manager().retrievers.get(file_hash)
        vector_store, bm25_shared = result if result else (None, None)

    faiss_ret = SessionManager.get("active_faiss_retriever", session_id=session_id)
    if not faiss_ret and vector_store:
        faiss_ret = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG.get("search_type", "similarity"),
            search_kwargs=RETRIEVER_CONFIG.get("search_kwargs", {"k": 5}),
        )
        SessionManager.set("active_faiss_retriever", faiss_ret, session_id=session_id)

    bm25_ret = SessionManager.get("active_bm25_retriever", session_id=session_id)
    if not bm25_ret and bm25_shared:
        bm25_ret = copy.copy(bm25_shared)
        SessionManager.set("active_bm25_retriever", bm25_ret, session_id=session_id)

    if bm25_ret:
        bm25_ret.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)

    return {
        "configurable": {
            "llm": llm,
            "session_id": session_id,
            "thread_id": session_id,
            "faiss_retriever": faiss_ret,
            "bm25_retriever": bm25_ret,
            "doc_language": SessionManager.get("doc_language", session_id=session_id),
        }
    }
