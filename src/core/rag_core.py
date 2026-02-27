"""
RAG 시스템의 통합 엔진 (Core Engine).
문서 로딩, 인덱싱, 검색, 질의응답의 모든 과정을 오케스트레이션합니다.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import logging
from typing import Any

import torch
from langchain_core.embeddings import Embeddings

from cache.vector_cache import VectorStoreCache
from common.config import ENABLE_VECTOR_CACHE, RETRIEVER_CONFIG
from common.exceptions import (
    EmptyPDFError,
    InsufficientChunksError,
    VectorStoreError,
)
from common.typing_utils import T
from core.chunking import split_documents
from core.document_processor import compute_file_hash, load_pdf_docs
from core.graph_builder import build_graph
from core.resource_pool import get_resource_pool
from core.retriever_factory import create_bm25_retriever, create_vector_store
from core.session import SessionManager
from services.optimization.index_optimizer import get_index_optimizer

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG 시스템의 통합 인터페이스.
    인덱싱부터 질의응답까지의 전체 라이프사이클을 관리합니다.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        SessionManager.init_session(session_id=session_id)

    def _ensure_session_context(self) -> None:
        """현재 스레드의 세션 컨텍스트를 보장합니다."""
        SessionManager.set_session_id(self.session_id)

    async def build_pipeline(
        self, file_path: str, file_name: str, embedder: Embeddings, on_progress=None
    ) -> tuple[str, bool]:
        """문서를 로드하고 RAG 파이프라인을 구축합니다."""
        self._ensure_session_context()

        file_hash = compute_file_hash(file_path)
        SessionManager.set("file_hash", file_hash, session_id=self.session_id)

        emb_model_name = getattr(
            embedder, "model", getattr(embedder, "model_name", "unknown")
        )
        cache = VectorStoreCache(file_path, emb_model_name, file_hash=file_hash)

        # 1. 캐시 시도
        if ENABLE_VECTOR_CACHE:
            cache_data = cache.load(embedder)
            if all(x is not None for x in cache_data):
                doc_splits, vector_store, bm25_retriever = cache_data
                SessionManager.add_status_log(
                    "⚡ 기존 분석 데이터 발견 (캐시 활용)", session_id=self.session_id
                )
                await self._register_and_finalize(
                    file_hash, vector_store, bm25_retriever, on_progress
                )
                return f"'{file_name}' 캐시 데이터 로드 완료", True

        # 2. 신규 문서 로드
        docs = load_pdf_docs(
            file_path, file_name, on_progress=on_progress, session_id=self.session_id
        )
        if not docs:
            raise EmptyPDFError(
                filename=file_name, details={"reason": "텍스트를 추출할 수 없습니다."}
            )

        # 3. 언어 감지
        sample_text = docs[0].page_content[:1000]
        lang = (
            "Korean"
            if any("\uac00" <= char <= "\ud7a3" for char in sample_text)
            else "English"
        )
        SessionManager.set("doc_language", lang, session_id=self.session_id)

        # 4. 청킹 및 벡터화
        doc_splits, vectors = await split_documents(
            docs, embedder, session_id=self.session_id
        )
        if not doc_splits:
            raise InsufficientChunksError(chunk_count=0, min_required=1)

        # 5. 인덱스 최적화 (선택적: 중복 제거 및 메타데이터 인덱싱)
        if sum(len(d.page_content) for d in docs) >= 2000 and vectors is not None:
            with contextlib.suppress(Exception):
                from services.optimization.index_optimizer import (
                    CompressionMethod,
                    IndexOptimizationConfig,
                    VectorQuantizationConfig,
                )

                # [최적화] 수동 양자화는 오버헤드가 크고 FAISS 내장 기능보다 비효율적이므로 NONE으로 설정
                opt_config = IndexOptimizationConfig(
                    quantization_config=VectorQuantizationConfig(
                        compression_method=CompressionMethod.NONE
                    )
                )
                optimizer = get_index_optimizer(opt_config)
                doc_splits, vectors, _, stats = optimizer.optimize_index(
                    doc_splits, vectors
                )
                SessionManager.add_status_log(
                    f"🧹 데이터 최적화 완료 ({stats.pruned_documents}개 중복 제거)",
                    session_id=self.session_id,
                )

        # 6. 컴포넌트 생성
        vector_store = create_vector_store(doc_splits, embedder, vectors=vectors)
        bm25_retriever = create_bm25_retriever(doc_splits)

        # 7. 캐시 저장
        if ENABLE_VECTOR_CACHE:
            cache.save(doc_splits, vector_store, bm25_retriever)

        # 8. 최종 등록
        await self._register_and_finalize(
            file_hash, vector_store, bm25_retriever, on_progress
        )

        # 메모리 정리
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        gc.collect()

        return f"'{file_name}' 신규 인덱싱 완료", False

    async def _register_and_finalize(
        self, file_hash, vector_store, bm25_retriever, on_progress
    ):
        """리소스를 등록하고 파이프라인을 최종 조립합니다."""
        await get_resource_pool().register(file_hash, vector_store, bm25_retriever)
        SessionManager.set("rag_engine", build_graph(), session_id=self.session_id)
        SessionManager.set("pdf_processed", True, session_id=self.session_id)
        SessionManager.add_status_log(
            "✨ 검색 엔진 구축 완료", session_id=self.session_id
        )
        if on_progress:
            on_progress()

    async def aquery(self, query: str, llm: T | None = None) -> dict[str, Any]:
        """[기본] 질문에 대한 답변을 비동기로 생성합니다 (Full Response)."""
        self._ensure_session_context()
        config = await self._prepare_config(llm)
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )
        return await rag_engine.ainvoke({"input": query}, config=config)

    async def astream_events(self, query: str, llm: T | None = None):
        """[스트리밍] 질문에 대한 이벤트를 발생시킵니다."""
        self._ensure_session_context()
        config = await self._prepare_config(llm)
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )
        return rag_engine.astream_events({"input": query}, config=config, version="v2")

    async def load_document(
        self, file_path: str, file_name: str, embedder: Embeddings, on_progress=None
    ) -> tuple[str, bool]:
        """build_pipeline의 하위 호환성 에일리어스"""
        return await self.build_pipeline(file_path, file_name, embedder, on_progress)

    async def _prepare_config(self, llm: T | None = None) -> dict:
        """검색기 및 모델 설정을 포함한 실행 Config를 준비합니다."""
        if llm:
            SessionManager.set("llm", llm, session_id=self.session_id)

        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        vector_store, bm25_shared = await get_resource_pool().get(file_hash)

        # 리소스 복구 로직
        if not vector_store:
            await self.build_pipeline(
                SessionManager.get("pdf_file_path", session_id=self.session_id),
                SessionManager.get(
                    "last_uploaded_file_name", session_id=self.session_id
                ),
                SessionManager.get("embedder", session_id=self.session_id),
            )
            vector_store, bm25_shared = await get_resource_pool().get(file_hash)

        faiss_ret = (
            vector_store.as_retriever(
                search_type=RETRIEVER_CONFIG["search_type"],
                search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
            )
            if vector_store
            else None
        )

        bm25_ret = copy.copy(bm25_shared) if bm25_shared else None
        if bm25_ret:
            bm25_ret.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)

        return {
            "configurable": {
                "llm": SessionManager.get("llm", session_id=self.session_id),
                "session_id": self.session_id,
                "faiss_retriever": faiss_ret,
                "bm25_retriever": bm25_ret,
                "doc_language": SessionManager.get(
                    "doc_language", session_id=self.session_id
                ),
            }
        }

    def get_status(self) -> list[str]:
        self._ensure_session_context()
        return SessionManager.get("status_logs", session_id=self.session_id) or []

    def clear_session(self) -> None:
        self._ensure_session_context()
        SessionManager.reset_all_state(session_id=self.session_id)
