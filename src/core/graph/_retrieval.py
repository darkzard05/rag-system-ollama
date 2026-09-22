"""검색 노드 단일 진원 (Phase 2A 병합).

구 ``_preprocess`` (의도 분류 + 쿼리 캐시 확인)와
구 ``_retrieve`` (하이브리드 검색 + 리랭킹 + 문맥 보강)를 흡수했다.
노드 I/O 스키마(반환 dict 키)는 변경 없이 유지하며,
구경로는 alias shim으로 유지한다.
"""

import asyncio
import copy
import logging
import time
from typing import Any, cast

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import AggregatedSearchResult, GraphState
from common.config import GRADING_CONFIG, QUERY_CACHE_ENABLED, QUERY_CACHE_MIN_CONF
from common.utils import doc_stable_id
from core.graph._graph_core import (
    _add_stage_ms,
    _ensure_query_cache_embedder,
    _enter_stage,
    _get_session_id,
    _reset_stage_timings,
    get_state_attr,
)
from core.session import SessionManager
from services.monitoring.performance_monitor import OperationType
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


# ============================================================================
# _preprocess 섹션 — 의도 분류 및 캐시 확인 노드 (pure move)
# ============================================================================


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
        file_hash = SessionManager.get("file_hash", session_id=sid, default=None)
        has_doc = bool(file_hash)
        if has_doc:
            await _ensure_query_cache_embedder()
            try:
                # D17 — cache key namespaced by file_hash so cross-document queries can't collide.
                cache_key = f"{file_hash}:{query}"
                cached = await get_cache_manager().get(cache_key, use_semantic=True)
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


# ============================================================================
# _retrieve 섹션 — 문서 검색 및 재순위화 노드 (pure move)
# ============================================================================

# 최종 컨텍스트에서 유지할 섹션의 최소 길이 임계값(문자 수).
# 이 값보다 짧은 섹션은 단편 청크로 판단되어 컨텍스트에서 제외된다.
# 50자 미만이면 대부분 "단락의 부제목만 추출된" 의미 없는 조각이므로
# generate 컨텍스트 오염을 막기 위해 드롭한다 (실측: `[GENERATE] 문서 0 길이: 43`).
_MIN_CONTEXT_SECTION_LEN = 50


def _filter_min_section_len(docs: list[Document]) -> list[Document]:
    """50자 미만 초단문 섹션을 최종 컨텍스트에서 제거합니다.

    임계값: ``_MIN_CONTEXT_SECTION_LEN`` (50자). 해당 길이 이상인 섹션만
    ``kept`` 로 보관된다.

    단답형 fallback 가드 (line 276 ``return kept if kept else docs[:1]``):
    모든 입력 문서가 50자 미만이라 ``kept`` 가 비어 있는 극단 케이스에서는
    컨텍스트가 완전히 비는 것을 막기 위해 원본에서 정확히 1개 문서만 유지한다.
    이 가드는 "무조건 1개 유지"가 아니라 "kept가 비었을 때만" 동작하므로,
    하나라도 50자 이상 문서가 있으면 그 문서들만 반환된다.
    """
    if not docs:
        return []
    kept = [d for d in docs if len(d.page_content) >= _MIN_CONTEXT_SECTION_LEN]
    return kept if kept else docs[:1]


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        return {}

    t0 = time.perf_counter()
    _retrieve_op = _enter_stage(OperationType.DOCUMENT_RETRIEVAL)
    _retrieve_op.__enter__()

    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    query = get_state_attr(state, "input")
    search_queries = get_state_attr(state, "search_queries")
    if search_queries:
        query = search_queries[-1]
        logger.debug(
            f"[RAG] [RETRIEVE] 재구성된 쿼리 사용: '{query}' (Retry: {get_state_attr(state, 'retry_count')})"
        )
        SessionManager.add_status_log(
            f"Retrying search with rewritten query: {query}",
            session_id=_get_session_id(config),
        )
    else:
        logger.debug(f"[RAG] [RETRIEVE] 원본 쿼리 기반 검색 시작: '{query}'")

    cfg = config.get("configurable", {})

    if writer is not None:
        await adispatch_custom_event(
            "graph_status",
            {"status": "관련 지식 검색 중..."},
            config=config,
        )
    SessionManager.add_status_log(
        f"Searching knowledge base: {query}", session_id=_get_session_id(config)
    )

    bm25 = cfg.get("bm25_retriever")
    faiss = cfg.get("faiss_retriever")

    from common.config import DYNAMIC_TOP_K_CONFIG, ENSEMBLE_WEIGHTS, RETRIEVER_CONFIG
    from core.retriever_factory import (
        search_bm25_with_scores,
        search_faiss_with_scores,
    )

    # [수정] R3a-01: 리트리버가 실제 반환한 점수를 메타데이터에서 캡처하도록
    # 점수 주입 검색 헬퍼를 사용한다 (FAISS similarity_search_with_score / BM25 get_top_n+get_scores).
    search_k = int(RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 25))

    search_tasks = {}
    if bm25:
        search_tasks["bm25"] = asyncio.create_task(
            search_bm25_with_scores(bm25, query, search_k)
        )
    if faiss:
        search_tasks["faiss"] = asyncio.create_task(
            search_faiss_with_scores(faiss, query, search_k)
        )

    results = {}
    if search_tasks:
        task_names = list(search_tasks.keys())
        task_results = await asyncio.gather(*search_tasks.values())
        results = dict(zip(task_names, task_results, strict=False))

    search_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        f"[RAG] [RETRIEVE] 검색 결과 확보 (BM25: {len(results.get('bm25', []))}, Vector: {len(results.get('faiss', []))})"
    )

    # [수정] R3a-01: 소스별 dict {"bm25": [...], "faiss": [...]}를 그대로 전달해
    # SearchResultAggregator._rrf_fusion_2node가 가중치와 소스별 순위를 실제 적용.
    # score=0.5 하드코딩 폴백 제거 — 리트리버 점수를 명시 캡처하고, 점수가 없는
    # 경우(테스트 대역 등)에만 순위 보존 폴백(0.0)을 사용한다.
    source_results: dict[str, list[AggregatedSearchResult]] = {}
    doc_map: dict[str, Document] = {}
    for source, res in results.items():
        node_results = []
        for doc in res:
            doc_id = doc_stable_id(doc)
            raw_score = doc.metadata.get("score")
            if raw_score is None:
                logger.debug(
                    f"[RAG] [RETRIEVE] {source} 소스 문서에 score 메타데이터 없음 — 순위 보존 폴백"
                )
                raw_score = 0.0
            node_results.append(
                AggregatedSearchResult(
                    doc_id=doc_id,
                    content=doc.page_content,
                    score=float(raw_score),
                    node_id=source,
                    metadata=doc.metadata,
                )
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        if node_results:
            source_results[source] = node_results

    aggregator = SearchResultAggregator()
    weights = get_state_attr(state, "search_weights") or {
        "bm25": ENSEMBLE_WEIGHTS[0],
        "faiss": ENSEMBLE_WEIGHTS[1],
    }

    # 실제 집계에 적용되는 가중치만 로그로 기록 (단일 소스 상황 포함)
    applied_desc = ", ".join(
        f"{nid}({weights.get(nid, 1.0):.2f})" for nid in source_results
    )
    logger.debug(f"[RAG] [RETRIEVE] 하이브리드 가중치 적용: {applied_desc}")

    aggregated, _ = aggregator.aggregate_results(
        source_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=search_k,
        weights=weights,
    )

    # [수정] R3a-04: 동적 Top-K 임계값을 RRF 점수 스케일(1/(k+rank))로 보정.
    # 가중치 합산 RRF의 상위권 최대 gap은 ~0.003 수준이므로 config 값(기본 0.003)으로 판정.
    dynamic_cfg = DYNAMIC_TOP_K_CONFIG
    gap_threshold = float(dynamic_cfg.get("gap_threshold", 0.003))
    min_candidates = int(dynamic_cfg.get("min_candidates", 12))
    max_candidates = int(dynamic_cfg.get("max_candidates", 18))

    if len(aggregated) >= 10:
        top_1_score = aggregated[0].aggregated_score
        top_10_score = aggregated[9].aggregated_score
        score_gap = top_1_score - top_10_score

        # 상위 그룹이 명확하면 후보군을 축소하고, 그렇지 않으면 최대 후보를 유지
        dynamic_top_k = min_candidates if score_gap > gap_threshold else max_candidates
        logger.debug(
            f"[RAG] [RETRIEVE] Dynamic Top-K 적용: {dynamic_top_k} "
            f"(Score Gap: {score_gap:.4f} / 임계값: {gap_threshold:.4f})"
        )
    else:
        dynamic_top_k = min_candidates

    # [R3b-02] RRF 집계 점수를 doc_id 기준으로 보관 — 6자 미만 쿼리 리랭킹 생략 경로에서
    # rerank_score로 기록해 grade short-circuit이 0.0으로만 평가되는 것을 방지한다.
    rrf_scores: dict[str, float] = {}
    final_docs = []
    for r in aggregated[:dynamic_top_k]:
        doc = doc_map.get(
            r.doc_id,
            Document(page_content=r.content, metadata=r.metadata),
        )
        rrf_scores[r.doc_id] = float(r.aggregated_score)
        final_docs.append(doc)

    aggregate_ms = (time.perf_counter() - t0) * 1000

    if not final_docs:
        q_len = len(query) if query else 0
        logger.warning(
            f"[RAG] [RETRIEVE] 검색 결과가 전혀 없습니다 (Query Length: {q_len})"
        )
        SessionManager.add_status_log(
            "No documents found.", session_id=_get_session_id(config)
        )
        return {"relevant_docs": []}

    from core.async_reranker import get_async_reranker

    reranker = await get_async_reranker()
    rerank_top_k = min(int(GRADING_CONFIG.get("top_k", 5)), len(final_docs))
    if len(query or "") < 6:
        ranked_docs = final_docs[:rerank_top_k]
        # [R3b-02] 6자 미만 쿼리는 리랭킹을 생략하되 rerank_score(RRF 집계 점수)를 기록해
        # grade short-circuit이 0.0으로만 평가되는 것을 방지한다.
        for doc in ranked_docs:
            doc_key = doc_stable_id(doc)
            doc.metadata["rerank_score"] = rrf_scores.get(doc_key, 0.0)
            doc.metadata["rerank_engine"] = "rrf"
    else:
        ranked_docs, _ = await reranker.rerank(
            final_docs,
            query=query,
            top_k=rerank_top_k,
        )
        from core.async_reranker import get_active_rerank_engine

        _engine = get_active_rerank_engine()
        for doc in ranked_docs:
            doc.metadata.setdefault("rerank_engine", _engine)
    rerank_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"[RAG] [RETRIEVE] 리랭킹 선별 완료: {len(final_docs)}개 후보 중 {len(ranked_docs)}개 최종 선별"
    )

    context_docs = cast(list[Document], ranked_docs)
    merged_context_docs = await asyncio.to_thread(
        _merge_adjacent_chunks, context_docs, max_tokens=800
    )
    # [T11] 50자 미만 초단문 섹션 드롭 — generate 컨텍스트가 단편 청크로 오염되지
    # 않도록 필터링한다. 백필 없음: 드롭으로 인한 문서 수 감소는 허용
    # (grade_top_n=3, top_k=5 하한 이상 유지).
    filtered_docs = _filter_min_section_len(merged_context_docs)
    dropped = len(merged_context_docs) - len(filtered_docs)
    if dropped > 0:
        logger.warning(
            "[RAG] [CTX] %d개 초단문 섹션 드롭 (< %d자)",
            dropped,
            _MIN_CONTEXT_SECTION_LEN,
        )
        SessionManager.add_status_log(
            f"{dropped}개 초단문 섹션(50자 미만)을 컨텍스트에서 제외했습니다.",
            session_id=_get_session_id(config),
        )
    merged_context_docs = filtered_docs
    logger.debug(
        f"[RAG] [RETRIEVE] 하이브리드 검색 및 문맥 보강 완료: 최종 {len(merged_context_docs)}개 섹션 구성"
    )

    merge_ms = (time.perf_counter() - t0) * 1000
    total_ms = merge_ms
    logger.debug(
        f"[RAG] [RETRIEVE][TIMING] search_ms={search_ms:.1f} "
        f"aggregate_ms={aggregate_ms:.1f} rerank_ms={rerank_ms:.1f} "
        f"merge_ms={merge_ms:.1f} total_ms={total_ms:.1f}"
    )

    _add_stage_ms("retrieve_ms", total_ms)
    _retrieve_op.__exit__(None, None, None)
    return {"relevant_docs": merged_context_docs}


def _merge_adjacent_chunks(
    docs: list[Document], max_tokens: int = 1200
) -> list[Document]:
    """같은 페이지의 연속된 청크들을 하나로 합쳐 풍부한 문맥을 제공합니다 (최적화 버전)."""
    if not docs:
        return []
    if len(docs) == 1:
        return docs

    from common.utils import count_tokens_rough

    merged_docs: list[Document] = []

    working_docs = sorted(
        docs,
        key=lambda x: (
            str(x.metadata.get("source", "")),
            int(x.metadata.get("page", 0)),
            int(x.metadata.get("chunk_index", 0)),
        ),
    )

    current_doc = Document(
        page_content=working_docs[0].page_content,
        metadata=copy.copy(working_docs[0].metadata),
    )

    current_tokens = count_tokens_rough(current_doc.page_content)

    for next_doc in working_docs[1:]:
        curr_m = current_doc.metadata
        next_m = next_doc.metadata

        is_same_context = curr_m.get("source") == next_m.get("source") and curr_m.get(
            "page"
        ) == next_m.get("page")

        is_same_section = curr_m.get("current_section") == next_m.get("current_section")

        curr_end = curr_m.get("end_index")
        next_start = next_m.get("start_index")

        if curr_end is not None and next_start is not None:
            is_actually_consecutive = abs(next_start - curr_end) <= 5
        else:
            is_actually_consecutive = (
                abs(next_m.get("chunk_index", 0) - curr_m.get("chunk_index", 0)) <= 1
            )

        next_tokens = count_tokens_rough(next_doc.page_content)

        if (
            is_same_context
            and is_actually_consecutive
            and is_same_section
            and (current_tokens + next_tokens + 10) <= max_tokens
        ):
            current_doc.page_content += "\n\n" + next_doc.page_content
            current_tokens += next_tokens + 10
            current_doc.metadata["end_index"] = next_m.get("end_index", curr_end)
            current_doc.metadata["chunk_index"] = next_m.get(
                "chunk_index", curr_m.get("chunk_index")
            )
            # [R3b-05] 병합 문서의 rerank_score는 head가 아닌 그룹 내 최대값으로 결정 —
            # grade short-circuit의 max_rerank_score와 UI 신뢰도 배지가 최고 관련 청크를 대표하도록 한다.
            curr_score = float(curr_m.get("rerank_score", 0.0) or 0.0)
            next_score = float(next_m.get("rerank_score", 0.0) or 0.0)
            current_doc.metadata["rerank_score"] = max(curr_score, next_score)
        else:
            merged_docs.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content,
                metadata=copy.copy(next_doc.metadata),
            )
            current_tokens = next_tokens

    merged_docs.append(current_doc)
    return merged_docs
