"""
RAG 파이프라인 구축·캐싱·쿼리 설정 로직.
RAGSystem에서 파이프라인 구축 및 실행 설정 책임을 분리하여 캡슐화합니다.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    RERANKER_MODEL_NAME,
    RETRIEVER_CONFIG,
    is_test_env,
)
from common.constants import GRADE_MEMO_KEY
from common.exceptions import EmptyPDFError, InsufficientChunksError, VectorStoreError
from core.build_state import PHASE_DETAIL_KEY, BuildEvent, BuildPhase
from core.chunking import split_documents
from core.document_processor import compute_file_hash, emit_progress, load_pdf_docs
from core.graph.graph_builder import build_graph, invalidate_graph_cache
from core.resource_manager import get_resource_manager
from core.retriever_factory import create_bm25_retriever, create_vector_store
from core.session import SessionManager
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


def _warmup_faiss_import() -> None:
    """FAISS/torch 첫회 import 비용을 선행 로드한다 (P3 cold 5s 회피).

    예외는 무시한다 — 실제 생성 시점에 정상 경로로 재시도된다.
    """
    try:
        import faiss  # noqa: F401
    except Exception as e:
        logger.debug(f"[RAG] [INDEX] FAISS 프리로드 스킵: {e}")
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            torch.cuda.get_device_properties(0)
    except Exception as e:
        logger.debug(f"[RAG] [INDEX] torch 프리로드 스킵: {e}")


# --- 모델 프리로드 (1회성, 비차단) ---
# _register_and_finalize 완료 시 기본 Ollama 모델을 백그라운드로 로드하여
# 첫 쿼리 시 LLM 콜드 스타트 지연을 완화합니다. 단일 인스턴스 스케줄링 상태는
# 테스트가 직접 리셋할 수 있도록 모듈 스칼라로 노출합니다.
_preload_lock: asyncio.Lock | None = None
_preload_loop: asyncio.AbstractEventLoop | None = None
_preload_scheduled: bool = False


async def _warmup_flashranker(ranker: Any) -> None:
    """FlashRank ONNX 세션을 실제 스코어링 1회로 확정합니다.

    ``get_flashranker()``는 파이썬 래퍼 객체만 반환하며, ONNX InferenceSession과
    가중치는 첫 ``CrossEncoder.rerank``(= predict) 호출에서 로드됩니다. 따라서
    acquire만으로는 첫 쿼리가 리랭킹 콜드 스타트를 그대로 지불합니다(측정 4.7s).

    FlashRank는 빈 passages를 받으면 즉시 반환하므로 1건 이상을 반드시 넘깁니다.
    """
    from flashrank import RerankRequest

    request = RerankRequest(query="warmup", passages=[{"id": 0, "text": "warmup"}])
    await asyncio.to_thread(ranker.rerank, request)


async def _preload_model() -> None:
    """임베딩·FlashRank·LLM을 프리로드합니다. 실패해도 앱을 중단하지 않으며 태스크 예외를 누출하지 않습니다."""
    try:
        from core.model_loader import ModelManager

        coordinator = get_resource_manager()

        # 1) 임베더 워밍 — 첫 쿼리 시 임베딩 모델 콜드 스타트(~10s) 방지.
        # embed_query는 블로킹 HTTP 호출이므로 to_thread로 실행해 이벤트 루프를 막지 않습니다.
        # 사용 기간 동안 use_embedder CM 으로 pin 하여 퇴출에 의한 use-after-free 방지.
        async with coordinator.use_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
            embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
            await asyncio.to_thread(embedder.embed_query, "warmup")
        logger.info("[RAG] [PRELOAD] 임베딩 워밍 완료")

        # 2) FlashRank 워밍 — 첫 쿼리 시 리랭커 lazy 로드(~10s) 방지.
        # acquire만으로는 ONNX InferenceSession이 뜨지 않으므로 실제 스코어링 1회가
        # 필요하다(측정: 첫 쿼리 rerank_ms=4744.8 → 이후 87.2).
        # FlashRank 실패(의존성 부재 포함)는 LLM 워밍까지 막지 않도록 여기서 격리한다.
        try:
            async with coordinator.use_flashranker(model_name=RERANKER_MODEL_NAME):
                ranker = await ModelManager.get_flashranker()
                await _warmup_flashranker(ranker)
            logger.info("[RAG] [PRELOAD] FlashRank 워밍 완료")
        except Exception as e:
            logger.warning(
                f"[RAG] [PRELOAD] FlashRank 워밍 실패 ({type(e).__name__}: {e}) — "
                "LLM 워밍은 계속 진행",
                exc_info=True,
            )

        # 3) LLM 프리로드 — 클라이언트 생성만으로는 Ollama 콜드 모델 로드가 첫 쿼리
        # 시점까지 지연되므로, 실질 워밍 추론을 1회 실행해 콜드 스타트를 선행합니다.
        # invoke는 블로킹 HTTP 호출이므로 to_thread로 실행해 이벤트 루프를 막지 않습니다.
        # 사용 기간 동안 use_llm CM 으로 pin 하여 퇴출에 의한 use-after-free 방지.
        async with coordinator.use_llm(model_name=DEFAULT_OLLAMA_MODEL):
            llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
            await asyncio.to_thread(llm.invoke, "warmup")
        logger.info("[RAG] [PRELOAD] LLM 워밍 추론 완료")
        logger.info(f"[RAG] [PRELOAD] 모델 프리로드 완료: {DEFAULT_OLLAMA_MODEL}")
    except Exception:
        logger.warning(
            f"[RAG] [PRELOAD] 모델 프리로드 실패: {DEFAULT_OLLAMA_MODEL}",
            exc_info=True,
        )


async def _schedule_model_preload() -> None:
    """모델 프리로드 태스크를 1회만 스케줄링합니다. (이벤트 루프 변경 시 락 재생성)"""
    # 테스트 환경(CI/유닛)에서는 pytest-asyncio가 테스트마다 이벤트 루프를 교체하므로
    # 프리로드 태스크가 루프 닫힘 시 모델별 Lock을 잡은 채 좌초되어 후속 테스트가
    # Lock.acquire에서 영원히 대기하는 데드락을 유발합니다.
    global _preload_lock, _preload_loop, _preload_scheduled
    if is_test_env():
        logger.info("[RAG] [PRELOAD] 테스트 환경 — 프리로드 스킵")
        return
    current_loop = asyncio.get_running_loop()
    # T1 unified rule (strategy A, best-effort variant): recreate on loop
    # change ONLY when idle. A foreign loop arriving while preload holds the
    # lock skips instead of forking (double warmup) or joining a foreign-bound
    # lock (RuntimeError on first contention) — preload is pure optimization.
    if _preload_lock is not None:
        if _preload_lock.locked() and _preload_loop is not current_loop:
            logger.debug("[RAG] [PRELOAD] 타 루프에서 프리로드 진행 중 — 스킵")
            return
        if _preload_loop is current_loop:
            lock = _preload_lock
        else:
            lock = asyncio.Lock()
            _preload_lock = lock
            _preload_loop = current_loop
    else:
        lock = asyncio.Lock()
        _preload_lock = lock
        _preload_loop = current_loop
    async with lock:
        if _preload_scheduled:
            return
        _preload_scheduled = True
        try:
            asyncio.create_task(_preload_model())
        except Exception:
            logger.warning("[RAG] [PRELOAD] 프리로드 태스크 생성 실패", exc_info=True)


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
        on_event: Callable[[BuildEvent], Any] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> tuple[str, bool]:
        """문서를 로드하고 RAG 파이프라인을 구축합니다.

        Args:
            file_path: PDF 파일 경로.
            file_name: 원본 파일 이름.
            embedder: 임베딩 모델 인스턴스.
            on_progress: 진행률 콜백 (0~100, 레거시 — P3에 제거 예정).
            check_cancelled: True 반환 시 asyncio.CancelledError 발생으로 중단.
            on_event: 타입 단계 이벤트 콜백 (신규 단일 계약).
            cancel: asyncio 취소 이벤트 (set 시 CancelledError로 중단).

        Returns:
            (메시지, 캐시 사용 여부) 튜플.
        """
        file_hash = compute_file_hash(file_path)
        prev_file_hash = SessionManager.get("file_hash", session_id=self.session_id)
        # [R1b-03] 팬텀 상태 방지: PDF 로드/청킹/벡터 빌드 등 실패 가능 작업
        # 이전의 file_hash 조기 커밋을 롤백으로 감쌉니다. 빌드 실패·취소 시
        # 이전 값으로 복원해 "새 해시 + 이전 엔진" 비정상 조합이 세션에 남아
        # 옛 문서 기준으로 동작하는 팬텀 상태가 생기지 않게 합니다.
        # [T18-회귀수정] pdf_file_path/파일명도 같은 커밋 단위로 기록·롤백합니다.
        # UI/API 업로드 경로는 이미 세션에 설정되지만, build_pipeline을 직접
        # 호출하는 경로(스크립트/테스트)는 비어 있어 쿼리 시점 퇴출 자기 치유
        # 재구축(get_or_build → build_fn)이 file_path=None으로 실패했습니다.
        prev_pdf_file_path = SessionManager.get(
            "pdf_file_path", session_id=self.session_id
        )
        prev_file_name = SessionManager.get(
            "last_uploaded_file_name", session_id=self.session_id
        )
        SessionManager.set("file_hash", file_hash, session_id=self.session_id)
        SessionManager.set("pdf_file_path", file_path, session_id=self.session_id)
        SessionManager.set(
            "last_uploaded_file_name", file_name, session_id=self.session_id
        )

        try:
            result = await self._build_impl(
                file_path,
                file_name,
                embedder,
                file_hash,
                on_progress,
                check_cancelled,
                on_event,
                cancel,
            )
        except BaseException:
            SessionManager.set("file_hash", prev_file_hash, session_id=self.session_id)
            SessionManager.set(
                "pdf_file_path", prev_pdf_file_path, session_id=self.session_id
            )
            SessionManager.set(
                "last_uploaded_file_name", prev_file_name, session_id=self.session_id
            )
            SessionManager.set("is_building_rag", False, session_id=self.session_id)
            logger.warning(
                "[RAG] [INDEX] 파이프라인 구축 실패 — file_hash 롤백: "
                f"{file_hash!r} -> {prev_file_hash!r}"
            )
            raise

        if (
            prev_file_hash
            and prev_file_hash != file_hash
            and prev_file_hash not in SessionManager.get_active_file_hashes()
        ):
            try:
                await get_resource_manager().unregister_retrievers(prev_file_hash)
            except Exception as e:
                logger.warning(f"[RAG] [INDEX] 이전 리트리버 정리 실패 — 우회: {e}")
            try:
                # 이전 문서의 좌표 캐시도 함께 정리 (신규 해시는 건드리지 않음:
                # 신규 좌표는 load_pdf_docs 단계에서 이미 저장됨).
                from cache.coord_cache import coord_cache

                await coord_cache.clear_cache(prev_file_hash)
            except Exception as e:
                logger.warning(f"[RAG] [CACHE] 이전 좌표 캐시 정리 실패 — 우회: {e}")
        return result

    async def _build_impl(
        self,
        file_path: str,
        file_name: str,
        embedder: Embeddings,
        file_hash: str,
        on_progress: Callable[[int], Any] | None,
        check_cancelled: Callable[[], bool] | None,
        on_event: Callable[[BuildEvent], Any] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> tuple[str, bool]:
        """file_hash 커밋 이후의 실제 파이프라인 구축 본문 (실패 가능 작업)."""

        def _check() -> None:
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError(
                    "파이프라인 구축이 사용자에 의해 취소되었습니다"
                )
            if check_cancelled is not None and check_cancelled():
                raise asyncio.CancelledError(
                    "파이프라인 구축이 사용자에 의해 취소되었습니다"
                )

        def _report(phase: BuildPhase, pct: int, status_text: str = "") -> None:
            """타입 이벤트 + 레거시 pct 콜백 이중 발행 (P3에 레거시 제거)."""
            if on_event is not None:
                try:
                    on_event(
                        BuildEvent(
                            phase=phase,
                            pct=pct,
                            detail_key=PHASE_DETAIL_KEY.get(phase, ""),
                            status=status_text,
                        )
                    )
                except Exception as e:
                    logger.warning(f"[RAG] [INDEX] on_event 콜백 실패 — 우회: {e}")
            emit_progress(on_progress, pct, detail=status_text)

        start_time = time.time()
        logger.debug(f"[RAG] [INDEX] 파이프라인 구축 시작: {file_name}")
        # Phase 0 계측: 단계별 wall-time 분리 (합계≒전체 검증용)
        _t_extract_ms = 0.0
        _t_chunk_ms = 0.0
        _t_faiss_ms = 0.0
        _t_bm25_ms = 0.0
        _t_register_ms = 0.0

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
                    logger.debug(
                        f"[RAG] [INDEX] 벡터 캐시 히트: {len(doc_splits)}개 청크 로드됨"
                    )

                    SessionManager.add_status_log(
                        "기존 분석 데이터 발견 (캐시 활용)", session_id=self.session_id
                    )
                    _report(BuildPhase.INDEXING, 90, "Loading from cache...")
                    await self._register_and_finalize(
                        file_hash,
                        vector_store,
                        bm25_retriever,
                        on_progress,
                        fresh_build=False,
                        on_event=on_event,
                    )
                    return f"'{file_name}' 캐시 데이터 로드 완료", True

        # 2. 신규 문서 로드
        SessionManager.add_status_log(
            f"'{file_name}' 텍스트 추출 중...", session_id=self.session_id
        )

        def _extract_progress(pct: int, detail: str = "") -> None:
            _report(BuildPhase.EXTRACTING, pct, detail or "Extracting text...")

        _t0 = time.perf_counter()
        # P3: FAISS/torch import를 추출과 중첩 — cold 첫회 5s를 크리티컬
        # 패스에서 제거한다 (추출 10s+와 오버랩).
        _faiss_warmup: asyncio.Task[None] | None = None
        try:
            _faiss_warmup = asyncio.create_task(asyncio.to_thread(_warmup_faiss_import))
        except RuntimeError:
            _faiss_warmup = None
        documents = await load_pdf_docs(
            file_path,
            file_name,
            on_progress=_extract_progress if on_event is not None else on_progress,
            session_id=self.session_id,
            file_hash=file_hash,
        )
        _t_extract_ms = (time.perf_counter() - _t0) * 1000.0
        if not documents:
            raise EmptyPDFError()

        _report(BuildPhase.EXTRACTING, 40, "Text extraction complete")

        # 3. 청크 분할 (Async)
        SessionManager.add_status_log(
            "문맥 최적화 및 청크 분할 중...", session_id=self.session_id
        )
        _report(BuildPhase.CHUNKING, 45, "Splitting documents...")
        _t0 = time.perf_counter()
        doc_splits, vectors = await split_documents(
            documents, embedder=embedder, session_id=self.session_id
        )
        _t_chunk_ms = (time.perf_counter() - _t0) * 1000.0
        if not doc_splits:
            raise InsufficientChunksError()

        _report(BuildPhase.CHUNKING, 55, "Chunking complete")

        _check()

        # 4–5. 벡터 스토어 및 BM25 리트리버 병렬 생성
        SessionManager.add_status_log(
            "지식 베이스(Vector Index) 생성 중...", session_id=self.session_id
        )
        _report(BuildPhase.INDEXING, 60, "Building vector index...")

        if _faiss_warmup is not None:
            # 추출과 중첩된 import 확정 (실패해도 생성 시점에 재시도됨)
            with contextlib.suppress(Exception):
                await _faiss_warmup

        async def _timed_faiss() -> Any:
            t0 = time.perf_counter()
            try:
                return await asyncio.to_thread(
                    create_vector_store,
                    doc_splits,
                    embedder,
                    vectors=vectors,
                    session_id=self.session_id,
                )
            finally:
                nonlocal _t_faiss_ms
                _t_faiss_ms = (time.perf_counter() - t0) * 1000.0

        async def _timed_bm25() -> Any:
            t0 = time.perf_counter()
            try:
                return await asyncio.to_thread(create_bm25_retriever, doc_splits)
            finally:
                nonlocal _t_bm25_ms
                _t_bm25_ms = (time.perf_counter() - t0) * 1000.0

        vector_store, bm25_retriever = await asyncio.gather(
            _timed_faiss(), _timed_bm25()
        )

        _report(BuildPhase.INDEXING, 68, "Vector index built")
        _report(BuildPhase.INDEXING, 75, "BM25 index built")

        _check()

        # 6. 등록 및 최종화 (캐시 저장 전에 수행 — 실패 시 캐시 미저장)
        _report(BuildPhase.REGISTERING, 80, "Registering retrievers...")
        _t0 = time.perf_counter()
        await self._register_and_finalize(
            file_hash,
            vector_store,
            bm25_retriever,
            on_progress,
            fresh_build=True,
            on_event=on_event,
        )
        _t_register_ms = (time.perf_counter() - _t0) * 1000.0

        # 7. 캐시 저장 (엔진 등록 성공 후에만 저장)
        if ENABLE_VECTOR_CACHE:
            cache.save(doc_splits, vector_store, bm25_retriever)

        duration = time.time() - start_time
        logger.info(
            f"[RAG] [INDEX] 파이프라인 구축 완료: {file_name} ({duration:.2f}s)"
        )
        logger.info(
            "[RAG] [INDEX][TIMING] extract_ms=%.1f chunk_ms=%.1f "
            "faiss_ms=%.1f bm25_ms=%.1f register_ms=%.1f total_s=%.2f chunks=%d",
            _t_extract_ms,
            _t_chunk_ms,
            _t_faiss_ms,
            _t_bm25_ms,
            _t_register_ms,
            duration,
            len(doc_splits),
        )
        return f"'{file_name}' 분석 및 신규 인덱싱 완료", False

    async def _register_and_finalize(
        self,
        file_hash: str,
        vector_store: Any,
        bm25_retriever: Any,
        on_progress: Callable[[int], Any] | None = None,
        fresh_build: bool = False,
        on_event: Callable[[BuildEvent], Any] | None = None,
    ) -> None:
        """생성된 리소스를 전역 풀에 등록하고 세션을 초기화합니다.

        fresh_build=True 이면 (캐시 재사용이 아닌) 실제 신규/교체 인덱싱이
        완료된 것이므로, 기존 코퍼스와 연결된 캐시/메모를 무효화합니다.
        """

        def _report(phase: BuildPhase, pct: int, status_text: str = "") -> None:
            if on_event is not None:
                try:
                    on_event(
                        BuildEvent(
                            phase=phase,
                            pct=pct,
                            detail_key=PHASE_DETAIL_KEY.get(phase, ""),
                            status=status_text,
                        )
                    )
                except Exception as e:
                    logger.warning(f"[RAG] [INDEX] on_event 콜백 실패 — 우회: {e}")
            emit_progress(on_progress, pct, detail=status_text)

        _report(BuildPhase.REGISTERING, 85, "Building graph workflow...")
        if fresh_build:
            # 신규/교체 인덱싱은 컴파일된 그래프를 재사용하면 구 코퍼스·구
            # 설정으로 고정된 stale 그래프가 등록된다. build_graph() 호출
            # 전에 캐시를 무효화해 이번 파이프라인용으로 재컴파일한다.
            # 캐시 재사용 경로(fresh_build=False)는 캐시 히트를 유지한다.
            invalidate_graph_cache()
        _, workflow = await asyncio.gather(
            get_resource_manager().register_retrievers(
                file_hash, vector_store, bm25_retriever
            ),
            build_graph(),
        )
        # 엔진과 file_hash 해시 메타데이터를 일관되게 캐싱합니다.
        # EngineCacheManager.get_engine이 해시 불일치(팬텀 상태)를 검출하게 합니다.
        # 지연 import: 단일 캐시 홈은 cache.vector_cache 이며,
        # 모듈 상단 import 는 rag_core 초기화 사이클(pipeline_builder->rag_core)을
        # 만들 수 있어 사용 시점에 가져온다.
        from cache.vector_cache import EngineCacheManager

        EngineCacheManager.set_engine(self.session_id, workflow)

        _report(BuildPhase.REGISTERING, 95, "Graph workflow ready")

        _report(BuildPhase.REGISTERING, 100, "Complete")

        # 신규 인덱싱 시에만 무효화: 캐시 재사용 빌드는 건드리지 않음.
        if fresh_build:
            await self._invalidate_caches_on_index()

        # 모델 프리로드 (1회성, 비차단 — 첫 쿼리 콜드 스타트 완화)
        await _schedule_model_preload()

    async def _invalidate_caches_on_index(self) -> None:
        """신규 인덱싱 후 쿼리 캐시·grade 메모 무효화 (실패해도 커밋 차단 안 함)."""
        # (1) 세맨틱 쿼리 캐시 무효화 (T5 대응: stale 답변 방지)
        try:
            cache_mgr = get_cache_manager()
            if cache_mgr.semantic_cache is not None:
                await cache_mgr.semantic_cache.clear()
                logger.info("[RAG] [CACHE] 신규 인덱싱 — 세맨틱 쿼리 캐시 무효화 완료")
        except Exception as e:
            logger.warning(f"[RAG] [CACHE] 무효화 실패 — 우회: {e}")

        # (2) 세션별 grade 결정 메모 무효화 (T8을 위한 계약 키 사용)
        try:
            SessionManager.delete(GRADE_MEMO_KEY, session_id=self.session_id)
            logger.info("[RAG] [CACHE] 신규 인덱싱 — grade 메모 무효화 완료")
        except Exception as e:
            logger.warning(f"[RAG] [CACHE] 무효화 실패 — 우회: {e}")


async def prepare_query_config_or_build(
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
    coordinator = get_resource_manager()

    result = coordinator.retrievers.get(file_hash)
    if result is None and pdf_file_path and build_fn is not None:
        # build_pipeline internally calls register_retrievers which
        # stores (vector_store, bm25) in coordinator.retrievers[file_hash].
        await build_fn(
            file_path=pdf_file_path,
            file_name=SessionManager.get(
                "last_uploaded_file_name", session_id=session_id
            ),
            embedder=embedder,
        )
        result = coordinator.retrievers.get(file_hash)
    pinned = False
    if result is not None:
        # [R1b-04] get_retrievers(내부 pin) 경유 — 진행 중인 세션의 리트리버를
        # LRU/용량 퇴출(silent eviction)에서 보호해 쿼리마다 재인덱싱되는
        # churn을 방지한다. 해제는 aquery/astream 종료 시 unpin_retrievers
        # (rag_core.py)가 담당한다 — 쿼리 스코프 pin이므로 조건 없이 해제해야
        # 하며, 단일 문서 보호는 RetrieverPool.check_memory_pressure가 담당한다.
        #
        # [T18-회귀수정] get_retrievers 내부 check_memory_pressure() await 사이
        # unpinned 리소스가 퇴출될 수 있다(메모리 압력 TOCTOU). build_fn을
        # get_or_build에 넘기면 build_pipeline의 반환값(메시지 문자열)이
        # 리소스로 저장되므로 부적합 — 퇴출로 인한 ValueError는 아래 기존
        # 빌드 폴백으로 자기 치유한다. build_fn이 없으면 None 폴백
        # ((None, None) 분기로 하류가 처리).
        try:
            result = await coordinator.get_retrievers(file_hash, None)
            pinned = True
        except ValueError:
            result = None
            if pdf_file_path and build_fn is not None:
                await build_fn(
                    file_path=pdf_file_path,
                    file_name=SessionManager.get(
                        "last_uploaded_file_name", session_id=session_id
                    ),
                    embedder=embedder,
                )
                result = coordinator.retrievers.get(file_hash)
                if result is not None:
                    # 재빌드 직후 비-pin get이므로 쿼리 스코프 보호를 위해 명시적 pin.
                    # 해제는 rag_core finally의 무조건 unpin이 담당한다.
                    coordinator.retrievers.pin(file_hash)
                    pinned = True
    try:
        vector_store, bm25_shared = result if result else (None, None)

        faiss_ret = SessionManager.get("active_faiss_retriever", session_id=session_id)
        bm25_ret = SessionManager.get("active_bm25_retriever", session_id=session_id)
        # [A1-self-heal] 세션 캐시 리트리버가 현재 file_hash와 다르면 폐기 후 재생성.
        # reset_for_new_file 누락·임베딩 변경 등 어떤 경로로 오염돼도 쿼리 설정 단계에서
        # 자기 치유된다. 해시 키는 파일 해시에서 파생된 별도 키로 약한 키와 충돌 방지.
        stored_retriever_hash = SessionManager.get(
            "active_retriever_hash", session_id=session_id
        )
        if stored_retriever_hash is not None and stored_retriever_hash != file_hash:
            SessionManager.set("active_faiss_retriever", None, session_id=session_id)
            SessionManager.set("active_bm25_retriever", None, session_id=session_id)
            SessionManager.delete("active_retriever_hash", session_id=session_id)
            faiss_ret = None
            bm25_ret = None
        if not faiss_ret and vector_store:
            faiss_ret = vector_store.as_retriever(
                search_type=RETRIEVER_CONFIG.get("search_type", "similarity"),
                search_kwargs=RETRIEVER_CONFIG.get("search_kwargs", {"k": 5}),
            )
            SessionManager.set(
                "active_faiss_retriever", faiss_ret, session_id=session_id
            )

        bm25_ret = SessionManager.get("active_bm25_retriever", session_id=session_id)
        if not bm25_ret and bm25_shared:
            bm25_ret = copy.copy(bm25_shared)
            SessionManager.set("active_bm25_retriever", bm25_ret, session_id=session_id)
        if (faiss_ret or bm25_ret) and file_hash:
            SessionManager.set(
                "active_retriever_hash", file_hash, session_id=session_id
            )

        if bm25_ret:
            bm25_ret.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)
    except Exception:
        # D1: pin 후 후속 단계(as_retriever 등)가 실패하면 호출자의
        # finally가 실행되지 않을 수 있으므로 여기서 선해제해 누수를 방지한다.
        # (이중 unpin은 BaseResourcePool.unpin이 흡수하므로 안전.)
        if pinned:
            coordinator.retrievers.unpin(file_hash)
        raise

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
