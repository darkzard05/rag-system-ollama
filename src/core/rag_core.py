"""
RAG 시스템의 통합 엔진 (Core Engine).
문서 로딩, 인덱싱, 검색, 질의응답의 모든 과정을 오케스트레이션합니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

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

        # 2. 신규 문서 로드 (Sync)
        SessionManager.add_status_log(
            f"'{file_name}' 텍스트 추출 중...", session_id=self.session_id
        )
        documents = load_pdf_docs(file_path, file_name, session_id=self.session_id)
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

        # 4. 벡터 스토어 생성 (Sync)
        SessionManager.add_status_log(
            "지식 베이스(Vector Index) 생성 중...", session_id=self.session_id
        )
        vector_store = create_vector_store(doc_splits, embedder, vectors=vectors)

        # 5. BM25 리트리버 생성 (Sync)
        bm25_retriever = create_bm25_retriever(doc_splits)

        # 6. 캐시 저장
        if ENABLE_VECTOR_CACHE:
            cache.save(doc_splits, vector_store, bm25_retriever)

        # 7. 등록 및 최종화
        await self._register_and_finalize(
            file_hash, vector_store, bm25_retriever, on_progress
        )

        duration = time.time() - start_time
        logger.info(
            f"[RAG] [INDEX] 파이프라인 구축 완료: {file_name} ({duration:.2f}s)"
        )
        return f"'{file_name}' 분석 및 신규 인덱싱 완료", False

    async def _register_and_finalize(
        self, file_hash, vector_store, bm25_retriever, on_progress=None
    ):
        """생성된 리소스를 전역 풀에 등록하고 세션을 초기화합니다."""
        await get_resource_pool().register(file_hash, vector_store, bm25_retriever)

        # 그래프 엔진 빌드
        workflow = build_graph()
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

    async def _get_rag_engine(self) -> Any:
        """
        현재 이벤트 루프에 적합한 RAG 엔진(LangGraph)을 반환합니다.
        이벤트 루프가 변경되었거나 엔진이 없으면 그래프를 즉시 컴파일하여 루프 불일치 예외를 방지합니다.
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        # 세션에서 기존 엔진과 루프 ID 가져오기
        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        cached_loop_id = SessionManager.get(
            "rag_engine_loop_id", 0, session_id=self.session_id
        )

        # 이벤트 루프 변경을 감지했을 때 즉시 LangGraph 컴파일러를 다시 호출합니다.
        if not rag_engine or cached_loop_id != current_loop_id:
            file_hash = SessionManager.get("file_hash", session_id=self.session_id)
            if not file_hash:
                return None

            from core.graph_builder import build_graph

            rag_engine = build_graph()

            SessionManager.set("rag_engine", rag_engine, session_id=self.session_id)
            SessionManager.set(
                "rag_engine_loop_id", current_loop_id, session_id=self.session_id
            )
            logger.info(
                f"[RAG] 비동기 루프 변경 감지({cached_loop_id} -> {current_loop_id}): LangGraph 엔진 핫스왑 재구성 성공"
            )

        return rag_engine

    async def aquery(self, query: str, model_name: str | None = None) -> dict[str, Any]:
        """[비동기] 질문에 대한 답변을 생성합니다 (전체 워크플로우 실행)."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)

        rag_engine = await self._get_rag_engine()
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        from services.monitoring.performance_monitor import (
            OperationType,
            get_performance_monitor,
        )

        monitor = get_performance_monitor()

        # [추가] 대화 이력 주입
        chat_history = self._get_recent_history()

        with monitor.track_operation(OperationType.RAG_PIPELINE_TOTAL):
            result = await rag_engine.ainvoke(
                {"input": query, "chat_history": chat_history}, config=config
            )

        docs = result.get("relevant_docs", [])
        await asyncio.to_thread(self._hydrate_docs, docs)

        from core.graph_builder import format_context

        # 성능 지표 통합
        perf_report = monitor.get_report()
        combined_perf = {**result.get("performance", {}), "metrics": perf_report}

        return {
            "response": result.get("response", ""),
            "thought": result.get("thought", ""),
            "context": format_context(docs),
            "documents": docs,
            "performance": combined_perf,
        }

    def _get_recent_history(self, limit: int = 5) -> list:
        """최근 대화 이력을 LangChain 메시지 객체 형식으로 변환하여 반환합니다."""
        from langchain_core.messages import AIMessage, HumanMessage

        raw_messages = SessionManager.get_messages(session_id=self.session_id)
        # 일반 대화(general)만 이력에 포함 (시스템 로그 등 제외)
        filtered = [m for m in raw_messages if m.get("msg_type") == "general"]
        recent = filtered[-limit:]

        formatted = []
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))
        return formatted

    def _hydrate_docs(self, docs: list[Document]) -> None:
        """문서 리스트의 좌표 데이터를 캐시에서 복구하거나, 없으면 즉시 추출(Lazy)합니다."""
        import fitz

        from cache.coord_cache import coord_cache

        # 1. 파일별로 처리 대상 문서 그룹화
        file_path_map: dict[str, list[Document]] = {}
        for doc in docs:
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
                with fitz.open(path) as doc_obj:
                    for doc in target_docs:
                        file_hash = doc.metadata.get("file_hash")
                        page_num = doc.metadata.get("page")

                        if not file_hash or page_num is None:
                            continue

                        coords = coord_cache.get_coords(file_hash, page_num)

                        if not coords:
                            logger.info(
                                f"[RAG] [HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} P{page_num}"
                            )
                            try:
                                page_obj = doc_obj[page_num - 1]

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
        """[스트리밍] astream_events(v2)를 사용하여 이벤트를 안전하게 발생시키고 브릿징합니다."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)

        rag_engine = await self._get_rag_engine()
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        # [추가] 대화 이력 주입
        chat_history = self._get_recent_history()

        async def _consumer():
            try:
                # astream_events(v2)는 multi-loop 및 nested 환경에서도 안정적인 이벤트 전파를 보장합니다.
                async for event in rag_engine.astream_events(
                    {"input": query, "chat_history": chat_history},
                    config=config,
                    version="v2",
                ):
                    kind = event["event"]
                    metadata = event.get("metadata", {})
                    langgraph_node = metadata.get("langgraph_node")

                    # 1. 커스텀 이벤트 브릿징 (generate 노드 등에서 발송)
                    if kind == "on_custom_event":
                        yield ("custom", event["data"])

                    # 2. 채팅 모델 스트림 브릿징 (폴백용)
                    # [최적화] generate 노드에서 직접 발생하는 스트림은 custom 이벤트를 통해 이미 전달되므로 중복 방지를 위해 스킵
                    elif kind == "on_chat_model_stream":
                        if langgraph_node == "generate":
                            continue
                        yield ("messages", event["data"])

                    # 3. 노드 업데이트 브릿징 (하이드레이션 처리 포함)
                    elif kind == "on_chain_stream":
                        data = event["data"]
                        chunk = data.get("chunk")
                        if chunk:
                            if isinstance(chunk, dict) and "retrieve" in chunk:
                                docs = chunk["retrieve"].get("relevant_docs", [])
                                if docs:
                                    # 비동기로 하이드레이션 시작 (블로킹 방지)
                                    asyncio.create_task(
                                        asyncio.to_thread(self._hydrate_docs, docs)
                                    )
                            yield ("updates", chunk)

            except Exception as e:
                logger.error(f"[RAG] 스트림 엔진 소비 중 에러: {e}", exc_info=True)
                raise e

        return _consumer()

    async def astream_events(self, query: str, model_name: str | None = None):
        """[스트리밍] 질문에 대한 이벤트를 발생시킵니다 (안전한 생산자-소비자 패턴)."""
        self._ensure_session_context()
        config = await self._prepare_config(model_name)

        rag_engine = await self._get_rag_engine()
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )

        import asyncio

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        producer_task: asyncio.Task | None = None

        async def _producer():
            try:
                async for event in rag_engine.astream_events(
                    {"input": query}, config=config, version="v2"
                ):
                    if event["event"] == "on_chain_stream":
                        docs = event["data"].get("chunk", {}).get("relevant_docs", [])
                        if docs:
                            await asyncio.to_thread(self._hydrate_docs, docs)

                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("[RAG] 큐 오버플로우: 소비자가 느림")
                        break
            except asyncio.CancelledError:
                logger.info("[RAG] 생산자 취소됨")
                raise
            except Exception as e:
                logger.error(f"[RAG] 생산자 오류: {e}", exc_info=True)
                await queue.put({"error": str(e)})
            finally:
                await queue.put(None)  # EOF 신호

        async def _consumer():
            nonlocal producer_task
            try:
                producer_task = asyncio.create_task(_producer())

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=300)
                    except asyncio.TimeoutError:
                        logger.warning("[RAG] 스트림 타임아웃 (5분)")
                        break

                    if event is None:
                        break

                    if isinstance(event, dict) and "error" in event:
                        logger.error(f"[RAG] 스트림 오류: {event['error']}")
                        break

                    yield event

            except asyncio.CancelledError:
                logger.info("[RAG] 소비자 취소됨")
                if producer_task:
                    producer_task.cancel()
                raise
            finally:
                if producer_task and not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass

        return _consumer()

    async def load_document(
        self, file_path: str, file_name: str, embedder: Embeddings, on_progress=None
    ) -> tuple[str, bool]:
        """build_pipeline의 하위 호환성 에일리어스"""
        return await self.build_pipeline(file_path, file_name, embedder, on_progress)

    async def _prepare_config(self, model_name: str | None = None) -> dict:
        """검색기 및 모델 설정을 포함한 실행 Config를 준비합니다."""
        from common.config import DEFAULT_OLLAMA_MODEL
        from core.model_loader import ModelManager

        target_model = model_name or DEFAULT_OLLAMA_MODEL
        llm = await ModelManager.get_llm(target_model)
        SessionManager.set("llm", llm, session_id=self.session_id)

        selected_embedding = SessionManager.get(
            "last_selected_embedding_model", session_id=self.session_id
        )
        if not selected_embedding:
            from common.config import DEFAULT_EMBEDDING_MODEL

            selected_embedding = DEFAULT_EMBEDDING_MODEL
        embedder = await ModelManager.get_embedder(selected_embedding)
        SessionManager.set("embedder", embedder, session_id=self.session_id)

        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        if not file_hash:
            raise VectorStoreError(details={"reason": "파일 해시 없음"})

        # 원자적 획득 또는 빌드 (중복 빌드 방지)
        pdf_file_path = SessionManager.get("pdf_file_path", session_id=self.session_id)
        if pdf_file_path:
            # 파이프라인이 필요한 경우에만 get_or_build 사용
            vector_store, bm25_shared = await get_resource_pool().get_or_build(
                file_hash,
                build_fn=self.build_pipeline,
                file_path=pdf_file_path,
                file_name=SessionManager.get(
                    "last_uploaded_file_name", session_id=self.session_id
                ),
                embedder=embedder,
            )
        else:
            # 파이프라인 불필요한 경우 단순 조회
            vector_store, bm25_shared = await get_resource_pool().get(file_hash)

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

        bm25_ret = SessionManager.get(
            "active_bm25_retriever", session_id=self.session_id
        )
        if not bm25_ret and bm25_shared:
            import copy

            bm25_ret = copy.copy(bm25_shared)
            SessionManager.set(
                "active_bm25_retriever", bm25_ret, session_id=self.session_id
            )

        if bm25_ret:
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
