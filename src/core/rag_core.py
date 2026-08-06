"""
RAG 시스템의 통합 오케스트레이터.
파이프라인 구축은 PipelineBuilder에, 문서 하이드레이션은 document_hydrator에,
쿼리 Config는 prepare_query_config에 위임합니다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from cache.engine_cache import EngineCacheManager
from common.circuit_breaker import get_circuit_breaker_registry
from common.exceptions import VectorStoreError
from core.document_hydrator import hydrate_documents
from core.pipeline_builder import PipelineBuilder, prepare_query_config_or_build
from core.resource_manager import get_resource_manager
from core.session import SessionManager

logger = logging.getLogger(__name__)


async def _stream_with_retry(
    event_stream_factory: Callable[[], AsyncIterator[dict]],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> AsyncIterator[dict]:
    for attempt in range(max_retries):
        yielded_any = False
        try:
            async for item in event_stream_factory():
                yielded_any = True
                yield item
            return
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            httpx.RequestError,
            httpx.TimeoutException,
        ) as e:
            if yielded_any:
                # 첫 토큰 이후 오류는 재시도하지 않는다 (중복 전송 방지).
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                f"[RAG] Stream retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise


def _log_hydration_task_failure(task: asyncio.Task[Any]) -> None:
    """완료 콜백: 실패한 하이드레이션 태스크의 예외를 소비·기록합니다.

    `t.exception()` 조회가 예외를 소비해 "Task exception was never retrieved"
    경고를 방지합니다. 예외를 재발생시키지 않습니다 (로그-온리 의미론).
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"[RAG] 문서 하이드레이션 실패: {exc}")


class RAGSystem:
    """RAG 오케스트레이터: 세션 라이프사이클 및 쿼리 흐름을 관리합니다."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        SessionManager.init_session(session_id=session_id)

    def _ensure_session_context(self) -> None:
        SessionManager.set_session_id(self.session_id)

    async def build_pipeline(
        self,
        file_path: str,
        file_name: str,
        embedder: Embeddings,
        on_progress: Callable[[int], Any] | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        self._ensure_session_context()
        builder = PipelineBuilder(session_id=self.session_id)
        return await builder.build(
            file_path=file_path,
            file_name=file_name,
            embedder=embedder,
            on_progress=on_progress,
            check_cancelled=check_cancelled,
        )

    async def _get_rag_engine(self) -> Any:
        rag_engine = EngineCacheManager.get_engine(self.session_id)
        if rag_engine:
            return rag_engine

        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        if not file_hash:
            return None
        from .graph_builder import build_graph

        rag_engine = await build_graph()
        EngineCacheManager.set_engine(self.session_id, rag_engine)
        return rag_engine

    async def aquery(self, query: str, model_name: str | None = None) -> dict[str, Any]:
        self._ensure_session_context()
        config = await prepare_query_config_or_build(
            self.session_id, model_name, build_fn=self.build_pipeline
        )
        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        try:
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
            chat_history = self._get_recent_history()
            with monitor.track_operation(OperationType.RAG_PIPELINE_TOTAL):
                result = await rag_engine.ainvoke(
                    {"input": query, "chat_history": chat_history}, config=config
                )
            docs = result.get("relevant_docs", [])
            await hydrate_documents(docs)
            from .graph_builder import format_context

            # perf_report = monitor.get_report() # 이전 코드
            perf_report = monitor.generate_report()
            combined_perf = {**result.get("performance", {}), "metrics": perf_report}
            return {
                "response": result.get("response", ""),
                "thought": result.get("thought", ""),
                "context": format_context(docs),
                "documents": docs,
                "performance": combined_perf,
            }
        finally:
            get_resource_manager().unpin_retrievers(file_hash)

    async def astream(self, query: str, model_name: str | None = None):
        self._ensure_session_context()
        config = await prepare_query_config_or_build(
            self.session_id, model_name, build_fn=self.build_pipeline
        )
        rag_engine = await self._get_rag_engine()
        if not rag_engine:
            raise VectorStoreError(
                details={"reason": "파이프라인이 준비되지 않았습니다."}
            )
        chat_history = self._get_recent_history()
        file_hash = SessionManager.get("file_hash", session_id=self.session_id)

        async def _consumer():
            # rewrite 루프가 retrieve 노드를 재실행하면 청크가 2회 발생하므로
            # 단일 변수가 아닌 리스트로 보관해 모든 하이드레이션 태스크를 유지한다.
            hydration_tasks: list[asyncio.Task[Any]] = []
            try:

                async def _event_factory():
                    breaker = get_circuit_breaker_registry().get_breaker(
                        "ollama",
                        session_id=self.session_id,
                        failure_threshold=5,
                        recovery_timeout=30,
                    )
                    async for event in breaker.call_async_stream(
                        rag_engine.astream_events,
                        {"input": query, "chat_history": chat_history},
                        config=config,
                        version="v2",
                    ):
                        yield event

                async for event in _stream_with_retry(_event_factory):
                    kind = event["event"]
                    metadata = event.get("metadata", {})
                    langgraph_node = metadata.get("langgraph_node")
                    if kind == "on_custom_event":
                        yield ("custom", event["data"])
                    elif kind == "on_chat_model_stream":
                        if langgraph_node == "generate":
                            continue
                        yield ("messages", event["data"])
                    elif kind == "on_chain_stream":
                        data = event["data"]
                        chunk = data.get("chunk")
                        if chunk:
                            if isinstance(chunk, dict) and "retrieve" in chunk:
                                docs = chunk["retrieve"].get("relevant_docs", [])
                                if docs:
                                    task = asyncio.create_task(hydrate_documents(docs))
                                    task.add_done_callback(_log_hydration_task_failure)
                                    hydration_tasks.append(task)
                            yield ("updates", chunk)
            except Exception as e:
                logger.error(f"[RAG] 스트림 에러: {e}", exc_info=True)
                raise e
            finally:
                # 하이드레이션은 await가 실패 태스크의 예외를 재발생시키므로
                # 반드시 try/except로 감싸 로그-온리 실패 의미론을 유지한다.
                # 완료 대기로 UI 경로의 _finalize_pdf_side_effects가
                # 좌표 완성 문서를 읽게 보장한다.
                for t in hydration_tasks:
                    try:
                        await t
                    except (Exception, asyncio.CancelledError) as te:
                        logger.error(f"[RAG] 문서 하이드레이션 실패: {te}")
                get_resource_manager().unpin_retrievers(file_hash)

        return _consumer()

    def _get_recent_history(self, limit: int = 5) -> list:
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

        raw_messages = SessionManager.get_messages(session_id=self.session_id)
        filtered = [m for m in raw_messages if m.get("msg_type") == "general"]
        formatted: list[BaseMessage] = []
        for msg in filtered[-limit:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))
        return formatted

    def get_status(self) -> list[str]:
        self._ensure_session_context()
        return SessionManager.get("status_logs", session_id=self.session_id) or []

    def clear_session(self) -> None:
        """현재 세션만 초기화합니다.

        전역 리소스 풀(모든 세션 공유)을 비우는 clear_all()은 다른 사용자의
        캐시를 파괴하므로 수행하지 않습니다. 현재 세션의 문서를 참조하는
        세션이 없을 때에만 해당 문서의 리트리버를 풀에서 제거합니다.

        답변 생성/스트리밍 중에는 초기화를 연기합니다. 도중에 삭제하면
        백그라운드 스트림 컨슈머의 청크가 새 세션 상태로 섞여 들어갑니다
        (교차 턴 오염).
        """
        self._ensure_session_context()
        if SessionManager.get(
            "is_generating_answer", False, session_id=self.session_id
        ):
            logger.warning(
                "[RAG] 답변 생성 중 세션 초기화 요청 — 스트림 종료 후 재시도하세요."
            )
            return

        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        SessionManager.reset_all_state(session_id=self.session_id)

        if file_hash:
            active_hashes = SessionManager.get_active_file_hashes()
            if file_hash not in active_hashes:
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        loop.create_task(
                            get_resource_manager().unregister_retrievers(file_hash)
                        )
                except RuntimeError:
                    logger.warning("[RAG] No running event loop for retriever cleanup")
