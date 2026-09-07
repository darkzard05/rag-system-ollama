"""의도 분류 및 캐시 확인 노드(``preprocess``).

``graph_builder.py`` 에서 추출됨 (graph-builder split plan Step 2).
``graph_builder.py`` 는 하위 호환을 위해 이 이름을 re-export 한다.
"""

import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import GraphState
from common.config import QUERY_CACHE_ENABLED, QUERY_CACHE_MIN_CONF
from core.graph._glue import _get_session_id
from core.graph._grading_glue import _add_stage_ms, _enter_stage, _reset_stage_timings
from core.graph._graph_utils import _ensure_query_cache_embedder, get_state_attr
from core.session import SessionManager
from services.monitoring.performance_monitor import OperationType
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


async def preprocess(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 수행합니다."""
    # [TIMING] 매 쿼리 시작 시 스테이지 누적 버퍼 초기화 (preprocess는 항상 최초 실행)
    _reset_stage_timings()
    _preprocess_op = _enter_stage(OperationType.QUERY_PROCESSING)
    _preprocess_op.__enter__()
    _preprocess_start = time.perf_counter()
    query = get_state_attr(state, "input", "").strip()
    logger.info(f"[RAG] [PREPROCESS] 입력 질의: '{query}'")

    import re

    from common.config import DYNAMIC_WEIGHTING_CONFIG, ENSEMBLE_WEIGHTS

    # 1. 의도 분류 및 동적 가중치 결정
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    intent = "rag"

    if len(query) < 10 and any(
        g in query.lower() for g in ["안녕", "hi", "hello", "반가워", "누구"]
    ):
        intent = "general"
        logger.info("[RAG] [PREPROCESS] 일상 대화(General) 의도 감지")

    if DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):
        keyword_patterns = DYNAMIC_WEIGHTING_CONFIG.get("keyword_patterns", [])
        is_keyword_heavy = any(re.search(p, query) for p in keyword_patterns)

        semantic_keywords = DYNAMIC_WEIGHTING_CONFIG.get("semantic_keywords", [])
        is_semantic_heavy = any(k in query for k in semantic_keywords)

        if is_keyword_heavy and not is_semantic_heavy:
            kw_w = DYNAMIC_WEIGHTING_CONFIG.get("keyword_weight", 0.8)
            weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}
            logger.info(f"[RAG] [PREPROCESS] 키워드 중심 질의 판단 (BM25: {kw_w})")
        elif is_semantic_heavy and not is_keyword_heavy:
            sm_w = DYNAMIC_WEIGHTING_CONFIG.get("semantic_weight", 0.8)
            weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}
            logger.info(f"[RAG] [PREPROCESS] 의미 중심 질의 판단 (FAISS: {sm_w})")

    _preprocess_ms = (time.perf_counter() - _preprocess_start) * 1000
    _add_stage_ms("preprocess_ms", _preprocess_ms)
    _preprocess_op.__exit__(None, None, None)

    # 쿼리 응답 캐시 조회 (선택적 opt-in). 세션에 활성 인덱싱 문서가 있을 때만
    # 후보군으로 삼는다 — 문서 없는 세션의 캐시 히트는 부정확하므로 무시한다.
    is_cached = False
    cached_response: str | None = None
    if QUERY_CACHE_ENABLED:
        sid = _get_session_id(config)
        has_doc = bool(SessionManager.get("file_hash", session_id=sid, default=None))
        if has_doc:
            await _ensure_query_cache_embedder()
            try:
                cached = await get_cache_manager().get(query, use_semantic=True)
            except Exception as e:  # noqa: BLE001 - 캐시 실패는 정상 경로로 폴백
                logger.warning(f"[RAG] [CACHE] 조회 실패 — 우회: {e}")
                cached = None
            if (
                isinstance(cached, dict)
                and isinstance(cached.get("response"), str)
                and cached[
                    "response"
                ].strip()  # 빈 문자열/공백/Nones는 히트로 취급하지 않음
                and float(cached.get("confidence", 0.0)) >= QUERY_CACHE_MIN_CONF
            ):
                response = cached["response"]
                is_cached = True
                cached_response = response
                intent = "general"  # 라우터가 generate로 단축 라우팅
                logger.info("[RAG] [PREPROCESS] 쿼리 캐시 히트 — generate 단축 경로")

    return {
        "intent": intent,
        "is_cached": is_cached,
        "cached_response": cached_response,
        "search_weights": weights,
        # 초단문 쿼리(<5자)는 그레이더 과해석(오타/타 문서에서의 환각) 위험이 커
        # strict 모드로 다룬다. grade_documents에서 활용한다.
        "short_query": len(query) < 5,
        # 턴 시작 시 이전 턴의 재작성 쿼리 잔재 제거 (reset_or_append 리듀서의 리셋 신호)
        "search_queries": [],
        # 턴 시작 시 이전 턴의 검색 문서 잔재 제거 (B7: retrieve_and_rerank/grade가
        # 이 키를 턴 간 누적/유지해 문서 없음 턴에서 이전 턴 문서가 프롬프트로 새는
        # cross-turn state leakage 방지). 빈 리스트로 교체해 격리한다.
        "relevant_docs": [],
        "retry_count": 0,
    }
