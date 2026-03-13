"""
RAG 시스템의 통합 엔진 (Core Engine).
문서 로딩, 인덱싱, 검색, 질의응답의 모든 과정을 오케스트레이션합니다.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from typing import Any

import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cache.vector_cache import VectorStoreCache
from common.config import ENABLE_VECTOR_CACHE, RETRIEVER_CONFIG
from common.exceptions import (
    EmptyPDFError,
    InsufficientChunksError,
    VectorStoreError,
)
from core.chunking import split_documents
from core.document_processor import compute_file_hash, load_pdf_docs
from core.graph_builder import build_graph
from core.resource_pool import get_resource_pool
from core.retriever_factory import create_bm25_retriever, create_vector_store
from core.session import SessionManager

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
        import time

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
        logger.info(f"[RAG] [INDEX] 문서 언어 감지: {lang}")

        # 4. 청킹 및 벡터화
        doc_splits, vectors = await split_documents(
            docs, embedder, session_id=self.session_id
        )
        if not doc_splits:
            raise InsufficientChunksError(chunk_count=0, min_required=1)

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

        duration = time.time() - start_time
        logger.info(
            f"[RAG] [INDEX] 신규 인덱싱 완료: {len(doc_splits)}개 청크 생성 (소요시간: {duration:.2f}s)"
        )

        # [복구] 메모리 정리
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
        SessionManager.add_status_log("검색 엔진 구축 완료", session_id=self.session_id)
        if on_progress:
            on_progress()

    async def aquery(self, query: str, model_name: str | None = None) -> dict[str, Any]:
        """[기본] 질문에 대한 답변을 비동기로 생성합니다 (Full Response)."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        # LangGraph 실행
        result = await rag_engine.ainvoke({"input": query}, config=config)

        # [추가] 좌표 데이터 복구 (Hydration)
        docs = result.get("relevant_docs", [])
        self._hydrate_docs(docs)

        from core.graph_builder import format_context

        return {
            "response": result.get("response", ""),
            "thought": result.get("thought", ""),
            "context": format_context(docs),
            "documents": docs,
            "performance": result.get("performance", {}),
        }

    def _hydrate_docs(self, docs: list[Document]) -> None:
        """문서 리스트의 좌표 데이터를 캐시에서 복구하거나, 없으면 즉시 추출(Lazy)합니다."""
        import fitz

        from cache.coord_cache import coord_cache

        # 1. 파일별로 처리 대상 문서 그룹화
        file_path_map: dict[str, list[Document]] = {}
        for doc in docs:
            # 이미 좌표가 있거나 메타데이터가 부족하면 스킵
            if "word_coords" in doc.metadata or not doc.metadata.get("has_coordinates"):
                continue

            path = doc.metadata.get("file_path")
            if path and os.path.exists(path):
                if path not in file_path_map:
                    file_path_map[path] = []
                file_path_map[path].append(doc)

        if not file_path_map:
            return

        # 2. 각 파일별로 1회만 열어서 모든 대상 문서(청크) 처리
        for path, target_docs in file_path_map.items():
            try:
                # [개선] Context Manager 사용으로 안전한 Close 보장
                with fitz.open(path) as doc_obj:
                    for doc in target_docs:
                        file_hash = doc.metadata.get("file_hash")
                        page_num = doc.metadata.get("page")

                        if not file_hash or page_num is None:
                            continue

                        # A. 캐시 확인
                        coords = coord_cache.get_coords(file_hash, page_num)

                        # B. 캐시 없으면 지연 추출 수행
                        if not coords:
                            logger.info(
                                f"[RAG] [HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} P{page_num}"
                            )
                            try:
                                page_obj = doc_obj[page_num - 1]

                                # [개선] 정밀 구역(Clip) 추출: 메타데이터의 bbox 영역만 타겟팅
                                chunk_bbox = doc.metadata.get("bbox")
                                if chunk_bbox:
                                    raw_words = page_obj.get_text(
                                        "words", clip=fitz.Rect(chunk_bbox)
                                    )
                                else:
                                    raw_words = page_obj.get_text("words")

                                coords = [
                                    (w[0], w[1], w[2], w[3], w[4]) for w in raw_words
                                ]

                                # 다음번을 위해 캐시 저장
                                coord_cache.save_coords(file_hash, page_num, coords)
                            except IndexError:
                                logger.warning(
                                    f"[RAG] [HYDRATE] 페이지 인덱스 초과: P{page_num}"
                                )
                                continue

                        if coords:
                            doc.metadata["word_coords"] = coords

            except Exception as e:
                logger.error(
                    f"[RAG] [HYDRATE] 파일 처리 중 오류 ({os.path.basename(path)}): {e}"
                )

    async def astream(self, query: str, model_name: str | None = None):
        """[스트리밍] 새로운 스트림 모드(messages, custom)를 사용하여 이벤트를 발생시킵니다."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        # [최적화] 익명 비동기 제너레이터를 반환하여 UI의 'await'와 호환성 유지
        async def _stream_wrapper():
            async for chunk in rag_engine.astream(
                {"input": query},
                config=config,
                stream_mode=["messages", "custom", "updates"],
            ):
                # 1. 메시지(messages) 모드 필터링: generate 노드의 메시지만 통과
                if (
                    isinstance(chunk, tuple)
                    and len(chunk) == 2
                    and chunk[0] == "messages"
                ):
                    msg, metadata = chunk[1]
                    # [표준화] LangGraph 메타데이터의 노드 이름 확인
                    node_name = metadata.get("langgraph_node")
                    # 'generate' 노드에서 생성된 메시지만 사용자에게 노출
                    if node_name != "generate":
                        continue

                # 2. 상태 업데이트(updates) 처리
                if isinstance(chunk, dict) and "retrieve" in chunk:
                    docs = chunk["retrieve"].get("relevant_docs", [])
                    self._hydrate_docs(docs)

                yield chunk

        return _stream_wrapper()

    async def astream_events(self, query: str, model_name: str | None = None):
        """[스트리밍] 질문에 대한 이벤트를 발생시킵니다 (레거시 adispatch_custom_event 대응)."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        # [최적화] 익명 비동기 제너레이터를 반환하여 UI의 'await'와 호환성 유지
        async def _event_wrapper():
            async for event in rag_engine.astream_events(
                {"input": query}, config=config, version="v2"
            ):
                # 'on_chain_stream' 이벤트 등에서 문서를 발견하면 복구
                if event["event"] == "on_chain_stream":
                    docs = event["data"].get("chunk", {}).get("relevant_docs", [])
                    if docs:
                        self._hydrate_docs(docs)

                yield event

        return _event_wrapper()

    async def load_document(
        self, file_path: str, file_name: str, embedder: Embeddings, on_progress=None
    ) -> tuple[str, bool]:
        """build_pipeline의 하위 호환성 에일리어스"""
        return await self.build_pipeline(file_path, file_name, embedder, on_progress)

    async def _prepare_config(self, model_name: str | None = None) -> dict:
        """검색기 및 모델 설정을 포함한 실행 Config를 준비합니다."""
        from common.config import DEFAULT_OLLAMA_MODEL
        from core.model_loader import ModelManager

        # 1. LLM 확보 (타입 안정성을 위해 기본값 명시)
        target_model = model_name or DEFAULT_OLLAMA_MODEL
        llm = await ModelManager.get_llm(target_model)
        SessionManager.set("llm", llm, session_id=self.session_id)

        # 2. 리소스 풀에서 리트리버 확보
        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        vector_store, bm25_shared = await get_resource_pool().get(file_hash)

        # 리소스 부재 시 복구 시도
        if not vector_store and SessionManager.get(
            "pdf_file_path", session_id=self.session_id
        ):
            logger.info(
                f"[RAG] 리소스 부재로 파이프라인 재구축 시도 (Hash: {file_hash[:8]})"
            )
            embedder = SessionManager.get("embedder", session_id=self.session_id)
            if embedder:
                await self.build_pipeline(
                    SessionManager.get("pdf_file_path", session_id=self.session_id),
                    SessionManager.get(
                        "last_uploaded_file_name", session_id=self.session_id
                    ),
                    embedder,
                )
                vector_store, bm25_shared = await get_resource_pool().get(file_hash)

        # 3. 개별 리트리버 인스턴스 구성 (세션 캐싱 활용)
        # [최적화] 매번 as_retriever를 호출하는 대신 세션에 저장하여 재사용
        faiss_ret = SessionManager.get(
            "active_faiss_retriever", session_id=self.session_id
        )
        if not faiss_ret and vector_store:
            faiss_ret = vector_store.as_retriever(
                search_type=RETRIEVER_CONFIG.get("search_type", "similarity"),
                search_kwargs=RETRIEVER_CONFIG.get("search_kwargs", {"k": 5}),
            )
            SessionManager.set(
                "active_faiss_retriever", faiss_ret, session_id=self.session_id
            )

        # [최적화] BM25 리트리버는 원본 인덱스는 공유하되 얕은 복사로 격리
        bm25_ret = SessionManager.get(
            "active_bm25_retriever", session_id=self.session_id
        )
        if not bm25_ret and bm25_shared:
            # 원본 객체를 직접 쓰지 않고 복사하여 파라미터 격리 (k 값 등)
            import copy

            bm25_ret = copy.copy(bm25_shared)
            SessionManager.set(
                "active_bm25_retriever", bm25_ret, session_id=self.session_id
            )

        if bm25_ret:
            # 설정 업데이트 (복사본이므로 안전함)
            target_k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)
            bm25_ret.k = target_k

        return {
            "configurable": {
                "llm": llm,
                "session_id": self.session_id,
                "thread_id": self.session_id,
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
