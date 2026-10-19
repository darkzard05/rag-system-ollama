"""
RAG 시스템의 통합 오케스트레이터.
파이프라인 구축은 PipelineBuilder에, 문서 하이드레이션은 document_hydrator에,
쿼리 Config는 prepare_query_config에 위임합니다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

from cache.vector_cache import EngineCacheManager
from common.config import (
    HYDRATION_ABORT_DRAIN_TIMEOUT_SECONDS,
    HYDRATION_TAIL_DRAIN_TIMEOUT_SECONDS,
)
from common.exceptions import VectorStoreError
from common.resilience import get_circuit_breaker_registry, retry_stream
from common.utils import doc_stable_id
from core.build_state import BuildEvent
from core.document_hydrator import hydrate_documents
from core.pipeline_builder import PipelineBuilder, prepare_query_config_or_build
from core.resource_manager import get_resource_manager
from core.session import SessionManager

logger = logging.getLogger(__name__)

# clear_session에서 생성한 정리 태스크 보관소. 지역 변수로 두면 이벤트 루프가
# 회수하기 전 태스크가 GC되어 "Task was destroyed but it is pending" 경고와
# 함께 정리가 중단될 수 있으므로 모듈 레벨 셋에 보관한다.
_pending_cleanup_tasks: set[asyncio.Task[Any]] = set()

# 정상 스트림 종료 후 tail 대기에서 포기했지만 취소하지 않은 하이드레이션
# 태스크 보관소. asyncio 이벤트 루프는 태스크를 약하게만 참조하므로, 강한
# 참조를 유지하지 않으면 "Task was destroyed but it is pending" 경고와 함께
# 좌표 추출이 중단될 수 있다. 완료 시 자동으로 제거된다.
_pending_hydration_tasks: set[asyncio.Task[Any]] = set()


@dataclass(frozen=True)
class HydrationDrainResult:
    """하이드레이션 드레인 결과 (태스크 순서 보존)."""

    results: list[Any] = field(default_factory=list)
    failed: int = 0
    timed_out: bool = False


async def _drain_hydration_tasks(
    tasks: list[asyncio.Task[Any]],
    *,
    timeout: float,
    msg_id: str = "",
    cancel_pending: bool = False,
    cancel_on_timeout: bool = True,
) -> HydrationDrainResult:
    """하이드레이션 태스크를 동시 대기한다 (순차 드레인 금지).

    ``asyncio.wait`` 단일 경로로 병렬 대기하며 결과 순서를 보존한다.
    태스크 실패는 세기만 하고 스트림 실패로 승격시키지 않는다. UI 턴 id는
    스트림까지 전달되지 않으므로 로그 상관 id는 호출자가 준다.

    ``cancel_pending=True``이면 대기 전에 pending을 먼저 취소한다(중단 경로).
    이미 포기된 턴의 좌표 작업을 백그라운드에서 계속 붙들지 않고 회수한다.

    ``cancel_on_timeout=True``(기본)이면 타임아웃 시 pending을 취소하고
    fail-loud로 알린다. ``False``이면 취소하지 않고 백그라운드에서 계속
    완료되게 두며 ``timed_out``만 표시한다(정상 스트림 tail 지연 방지).
    태스크는 호출자가 강한 참조로 보관해 GC되지 않게 해야 한다.
    """
    if cancel_pending:
        for t in tasks:
            if not t.done():
                t.cancel()
    results: list[Any] = [None] * len(tasks)
    failed = 0
    timed_out = False
    _t0 = time.perf_counter()
    pending = [t for t in tasks if not t.done()]
    if pending:
        index_of = {id(t): i for i, t in enumerate(tasks)}
        try:
            done, still_pending = await asyncio.wait(pending, timeout=timeout)
        except asyncio.CancelledError:
            # 바깥 취소: 태스크를 방치하지 않고 함께 회수한 뒤 재발생.
            for t in pending:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        if still_pending:
            timed_out = True
            if cancel_on_timeout:
                for t in still_pending:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*still_pending, return_exceptions=True)
                for t in still_pending:
                    if t.cancelled() or t.exception() is not None:
                        failed += 1
                logger.error("[RAG] 하이드레이션 드레인 타임아웃")
            else:
                logger.info(
                    "[RAG] 하이드레이션 드레인 타임아웃 — %d개 태스크 "
                    "백그라운드 계속 msg_id=%s",
                    len(still_pending),
                    msg_id or "-",
                )
        for task in done:
            if task.cancelled():
                failed += 1
                continue
            exc = task.exception()
            if exc is not None:
                failed += 1
                logger.error(f"[RAG] 문서 하이드레이션 실패: {exc}")
                continue
            results[index_of[id(task)]] = task.result()
    elapsed = time.perf_counter() - _t0
    logger.info(
        "[PERF] hydration drain took %.3fs (%d task(s), waited=%s skipped_cache_hit=%s) msg_id=%s",
        elapsed,
        len(tasks),
        len(pending) > 0,
        len(pending) == 0,
        msg_id or "-",
    )
    return HydrationDrainResult(results=results, failed=failed, timed_out=timed_out)


def _lazy_perf_report() -> dict[str, Any]:
    """모니터링 스냅샷을 best-effort로 구한다 (답변 경로를 깨지 않는다)."""
    try:
        from services.monitoring.performance_monitor import (
            get_performance_monitor,
        )

        report = get_performance_monitor().generate_report()
        return dict(report) if isinstance(report, dict) else {}
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("[PERF] perf report skipped (lazy): %s", exc)
        return {}


async def _stream_with_retry(
    event_stream_factory: Callable[[], AsyncIterator[dict]],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> AsyncIterator[dict]:
    # R12: 원래의 인라인 재시도 루프를 common/retry.py 의 retry_stream 으로
    # 위임한다. 정확한 의미론(base_delay=1.0, 2**attempt 백오프, "첫 토큰 이후
    # 재시도 안 함", 동일 예외 집합, asyncio.CancelledError 재발생)을 보존한다.
    async for item in retry_stream(
        event_stream_factory,
        max_retries=max_retries,
        base_delay=base_delay,
    ):
        yield item


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


def _dedup_docs(docs: list, seen: set[str]) -> list:
    """rewrite 루프 재실행 시 중복 도착 문서를 제거합니다.

    `seen`은 호출자(_consumer)가 소유하며 호출 간 유지됩니다.
    안정 키는 doc_id 메타데이터 우선, 없으면 page_content 해시로 계산합니다.
    """
    kept: list = []
    for _d in docs:
        _key = doc_stable_id(_d)
        if _key in seen:
            continue
        seen.add(_key)
        kept.append(_d)
    return kept


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
        on_event: Callable[[BuildEvent], Any] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> tuple[str, bool]:
        self._ensure_session_context()
        builder = PipelineBuilder(session_id=self.session_id)
        return await builder.build(
            file_path=file_path,
            file_name=file_name,
            embedder=embedder,
            on_progress=on_progress,
            check_cancelled=check_cancelled,
            on_event=on_event,
            cancel=cancel,
        )

    async def _get_rag_engine(self) -> Any:
        rag_engine = EngineCacheManager.get_engine(self.session_id)
        if rag_engine:
            return rag_engine

        file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        if not file_hash:
            return None
        from .graph.graph_builder import build_graph

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
            # R1a-04: chat_history 채널 제거 — 히스토리는 SessionManager가 전담한다.
            with monitor.track_operation(OperationType.RAG_PIPELINE_TOTAL):
                result = await rag_engine.ainvoke({"input": query}, config=config)
            docs = result.get("relevant_docs", [])
            await hydrate_documents(docs)
            from .graph.graph_builder import format_context, resolve_cached_context

            # CTX 재사용: 그래프 generate이 상태에 기록한 context를 우선 사용해
            # 동일 docs의 3차 렌더·재해시를 건너뛴다. 없으면 기존대로 렌더.
            # (하이드레이션이 추가하는 좌표 메타데이터는 렌더 필드에 무영향.)
            cached_context = resolve_cached_context(result, docs)

            # perf_report = monitor.get_report() # 이전 코드
            perf_report = _lazy_perf_report()
            combined_perf = {**result.get("performance", {}), "metrics": perf_report}
            return {
                "response": result.get("response", ""),
                "thought": result.get("thought", ""),
                "context": cached_context
                if cached_context is not None
                else format_context(docs),
                "documents": docs,
                "performance": combined_perf,
            }
        finally:
            # D1: 쿼리 스코프 pin 균형 — prepare 단계의 get_retrievers pin(+1)은
            # 쿼리마다 반드시 해제한다. 단일 문서 보호는
            # RetrieverPool.check_memory_pressure(len<=1 no-evict)가 담당하므로
            # 여기서 조건부로 유지하면 pin refcount가 쿼리 수만큼 누적되어
            # clear_session의 unregister가 영원히 실패한다.
            if file_hash:
                get_resource_manager().unpin_retrievers(file_hash)

    async def astream(self, query: str, model_name: str | None = None):
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
        except Exception:
            if file_hash:
                get_resource_manager().unpin_retrievers(file_hash)
            raise

        async def _consumer():
            # rewrite 루프가 retrieve 노드를 재실행하면 청크가 2회 발생하므로
            # 단일 변수가 아닌 리스트로 보관해 모든 하이드레이션 태스크를 유지한다.
            hydration_tasks: list[asyncio.Task[Any]] = []
            _aborted = False
            seen_doc_keys: set[str] = set()
            # Tier1 P0-2: generate 토큰 드롭 합류점 계측. custom response_chunk가
            # 정상 경로이며 generate 노드 messages는 중복 방지용으로 드롭한다.
            # writer=None·파서 폴백 등으로 custom이 없으면 무응답이 되므로,
            # 드롭 수와 custom 수신을 세어 침묵 실패를 로그로 가시화한다.
            dropped_generate_msgs = 0
            custom_events = 0
            SessionManager.add_status_log(
                "답변 생성 시작...",
                session_id=self.session_id,
            )
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
                        # R1a-04: chat_history 채널 제거 — 히스토리는 SessionManager가 전담한다.
                        {"input": query},
                        config=config,
                        version="v2",
                    ):
                        yield event

                async for event in _stream_with_retry(_event_factory):
                    kind = event["event"]
                    metadata = event.get("metadata", {})
                    langgraph_node = metadata.get("langgraph_node")
                    if kind == "on_custom_event":
                        custom_events += 1
                        yield ("custom", event["data"])
                    elif kind == "on_chat_model_stream":
                        if langgraph_node == "generate":
                            dropped_generate_msgs += 1
                            continue
                        yield ("messages", event["data"])
                    elif kind == "on_chain_stream":
                        data = event["data"]
                        chunk = data.get("chunk")
                        if chunk:
                            if isinstance(chunk, dict) and "retrieve" in chunk:
                                docs = chunk["retrieve"].get("relevant_docs", [])
                                if docs:
                                    new_docs = _dedup_docs(docs, seen_doc_keys)
                                    if new_docs:
                                        task = asyncio.create_task(
                                            hydrate_documents(new_docs)
                                        )
                                        _pending_hydration_tasks.add(task)
                                        task.add_done_callback(
                                            _pending_hydration_tasks.discard
                                        )
                                        task.add_done_callback(
                                            _log_hydration_task_failure
                                        )
                                        hydration_tasks.append(task)
                            yield ("updates", chunk)
            except (asyncio.CancelledError, GeneratorExit):
                # 중단(클라이언트 절단·waiter 취소·aclose) — finally에서
                # 하이드레이션을 즉시 취소하고 짧게 드레인한다.
                _aborted = True
                raise
            except Exception as e:
                logger.error(f"[RAG] 스트림 에러: {e}", exc_info=True)
                raise e
            finally:
                if dropped_generate_msgs > 0 and custom_events == 0:
                    # custom 경로 없이 generate 토큰만 드롭됨 — 무응답 후보.
                    # 동작은 유지(중복 방지)하고 경고로 후속 재현을 유도한다.
                    logger.warning(
                        "[RAG] generate 토큰 드롭됨 — custom 이벤트 없음 "
                        f"(dropped={dropped_generate_msgs}, session={self.session_id})"
                    )
                    SessionManager.add_status_log(
                        "답변 스트림이 비어 있습니다 (generate 폴백 미수신).",
                        session_id=self.session_id,
                    )
                # [최적화] 텍스트 스트리밍 완료 후 UI 응답 차단을 막기 위해
                # 정상 경로는 짧은 tail 상한만 기다리고, 미완료 좌표 작업은
                # 취소하지 않고 백그라운드에서 계속 완료되게 둔다. 중단 경로는
                # 이미 포기된 턴이므로 기존처럼 즉시 취소·회수한다.
                drain = await _drain_hydration_tasks(
                    hydration_tasks,
                    timeout=(
                        HYDRATION_ABORT_DRAIN_TIMEOUT_SECONDS
                        if _aborted
                        else HYDRATION_TAIL_DRAIN_TIMEOUT_SECONDS
                    ),
                    msg_id=self.session_id,
                    cancel_pending=_aborted,
                    cancel_on_timeout=_aborted,
                )
                if not _aborted and drain.timed_out:
                    logger.debug(
                        "[RAG] 하이드레이션 백그라운드 연속 처리 진행 "
                        "(Tail Latency 방지: msg_id=%s)",
                        self.session_id,
                    )
                # 중단/백그라운드 계속은 실패가 아니므로, 실제로 완료된 태스크의
                # 실패만 상태 로그로 표면화한다.
                if not _aborted and drain.failed:
                    SessionManager.add_status_log(
                        "일부 참조 문서의 하이라이트 좌표를 불러오지 못했습니다.",
                        session_id=self.session_id,
                    )
                # D1: 쿼리 스코프 pin 균형 — prepare 단계의 pin(+1)은 스트림 종료 시
                # 반드시 해제한다 (단일 문서 no-evict는 check_memory_pressure가 담당).
                if file_hash:
                    get_resource_manager().unpin_retrievers(file_hash)

        return _consumer()

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

        # T1: 세션 스코프 그래프 상태(체크포인터 thread + speculative orphan)를
        # 명시적으로 정리한다. delete_session 경로에서도 수행되지만, clear 경로의
        # 정리 계약을 이 메서드에 직접 둔다 (멱등 — 없어도 조용히 무시).
        try:
            from core.graph._speculative_gen import delete_spec_thread
            from core.graph.graph_builder import delete_graph_thread

            delete_spec_thread(self.session_id)
            delete_graph_thread(self.session_id)
        except Exception as e:
            logger.warning(f"[RAG] 세션 그래프 상태 정리 실패 ({self.session_id}): {e}")

        if file_hash:
            active_hashes = SessionManager.get_active_file_hashes()
            if file_hash not in active_hashes:
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        task = loop.create_task(
                            get_resource_manager().unregister_retrievers(file_hash)
                        )

                        def _cleanup_done(
                            t: asyncio.Task[Any], _hash: str = file_hash
                        ) -> None:
                            _pending_cleanup_tasks.discard(t)
                            try:
                                removed = t.result()
                            except Exception as e:
                                logger.warning(
                                    f"[RAG] 리트리버 정리 실패 ({_hash}): {e}"
                                )
                            else:
                                if removed is False:
                                    logger.warning(
                                        f"[RAG] 리트리버 정리 연기 "
                                        f"({_hash}: 사용 중 pin)"
                                    )

                        _pending_cleanup_tasks.add(task)
                        task.add_done_callback(_cleanup_done)
                except RuntimeError:
                    logger.warning("[RAG] No running event loop for retriever cleanup")
