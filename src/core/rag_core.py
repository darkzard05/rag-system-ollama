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
        try:
            async for item in event_stream_factory():
                yield item
            return
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            httpx.RequestError,
            httpx.TimeoutException,
        ) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                f"[RAG] Stream retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise


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
                                    try:
                                        await hydrate_documents(docs)
                                    except Exception as task_e:
                                        logger.error(
                                            f"[RAG] 문서 하이드레이션 실패: {task_e}"
                                        )
                            yield ("updates", chunk)
            except Exception as e:
                logger.error(f"[RAG] 스트림 에러: {e}", exc_info=True)
                raise e
            finally:
                get_resource_manager().unpin_retrievers(file_hash)

        return _consumer()

    def _get_recent_history(self, limit: int = 5) -> list:
        from langchain_core.messages import AIMessage, HumanMessage

        raw_messages = SessionManager.get_messages(session_id=self.session_id)
        filtered = [m for m in raw_messages if m.get("msg_type") == "general"]
        formatted = []
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
        self._ensure_session_context()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(get_resource_manager().clear_all())
        except RuntimeError:
            logger.warning("[RAG] No running event loop for resource pool cleanup")
        SessionManager.reset_all_state(session_id=self.session_id)
