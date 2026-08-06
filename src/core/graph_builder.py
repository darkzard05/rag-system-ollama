"""
LangGraph를 사용하여 자가 교정(Self-Correction) RAG 워크플로우를 구성합니다.
의도 분류, 캐시 확인, 하이브리드 검색, 문서 평가, 쿼리 재구성, 생성의 단계를 포함합니다.
"""

import asyncio
import copy
import logging
import warnings
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from api.schemas import AggregatedSearchResult, GraphState
from common.config import (
    ANALYSIS_PROTOCOL,
    GRADING_CONFIG,
)
from common.utils import fast_hash
from core.model_loader import ModelManager
from core.session import SessionManager

logger = logging.getLogger(__name__)


def _get_session_id(config: RunnableConfig | None = None) -> str:
    """Extract session_id from RunnableConfig or fall back to current context."""
    if config and "configurable" in config:
        sid = config["configurable"].get("session_id")
        if sid:
            return sid
        # config는 전달됐지만 session_id가 누락된 경우 — 전파 버그 신호.
        # 정상적인 "default" 세션 사용(비 Streamlit 모드)은 조용히 폴백합니다.
        logger.warning("[GRAPH] config에 session_id 누락 — 암묵적 세션 폴백")
    return SessionManager.get_session_id()


class _GraphCache:
    """컴파일된 그래프와 빌드 락을 안전하게 캡슐화합니다."""

    def __init__(self) -> None:
        self._compiled: Any = None
        self._lock: asyncio.Lock | None = None

    def get_lock(self) -> asyncio.Lock:
        """지연 초기화된 그래프 빌드 락을 반환합니다."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def compiled(self) -> Any:
        return self._compiled

    @compiled.setter
    def compiled(self, value: Any) -> None:
        self._compiled = value

    def invalidate(self) -> None:
        """컴파일된 그래프를 무효화하여 다음 build_graph() 호출 시 재컴파일합니다."""
        self._compiled = None


_graph_cache = _GraphCache()


def invalidate_graph_cache() -> None:
    """Force recompilation of the LangGraph on the next build_graph() call."""
    _graph_cache.invalidate()


def get_state_attr(state: Any, key: str, default: Any = None) -> Any:
    """dict와 object(GraphState) 모두에서 속성을 안전하게 가져옵니다."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


async def _safe_invoke(llm: Any, prompt: str, config: dict) -> Any:
    """Safely invoke an LLM with async fallback."""
    res = llm.ainvoke(prompt, config=config)
    if asyncio.iscoroutine(res):
        return await res
    return res


async def preprocess(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 수행합니다."""
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

    return {
        "intent": intent,
        "is_cached": False,
        "search_weights": weights,
        "retry_count": 0,
    }


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        return {}

    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    query = get_state_attr(state, "input")
    search_queries = get_state_attr(state, "search_queries")
    if search_queries:
        query = search_queries[-1]
        logger.info(
            f"[RAG] [RETRIEVE] 재구성된 쿼리 사용: '{query}' (Retry: {get_state_attr(state, 'retry_count')})"
        )
        SessionManager.add_status_log(
            f"재구성된 쿼리로 검색 시도: {query}", session_id=_get_session_id(config)
        )
    else:
        logger.info(f"[RAG] [RETRIEVE] 원본 쿼리 기반 검색 시작: '{query}'")

    cfg = config.get("configurable", {})

    if writer is not None:
        await adispatch_custom_event(
            "graph_status", {"status": "관련 지식 탐색 중..."}, config=config
        )
    SessionManager.add_status_log(
        f"지식 탐색 중: {query}", session_id=_get_session_id(config)
    )

    bm25 = cfg.get("bm25_retriever")
    faiss = cfg.get("faiss_retriever")

    search_tasks = {}
    if bm25:
        search_tasks["bm25"] = asyncio.create_task(bm25.ainvoke(query))
    if faiss:
        search_tasks["faiss"] = asyncio.create_task(faiss.ainvoke(query))

    results = {}
    if search_tasks:
        task_names = list(search_tasks.keys())
        task_results = await asyncio.gather(*search_tasks.values())
        results = dict(zip(task_names, task_results, strict=False))

    logger.debug(
        f"[RAG] [RETRIEVE] 검색 결과 확보 (BM25: {len(results.get('bm25', []))}, Vector: {len(results.get('faiss', []))})"
    )

    all_docs = []
    doc_map: dict[str, Document] = {}
    for source, res in results.items():
        for doc in res:
            doc_id = doc.metadata.get("doc_id", fast_hash(doc.page_content))
            all_docs.append(
                AggregatedSearchResult(
                    doc_id=doc_id,
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.5),
                    node_id=source,
                    metadata=doc.metadata,
                )
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    from common.config import ENSEMBLE_WEIGHTS

    aggregator = SearchResultAggregator()
    weights = get_state_attr(state, "search_weights") or {
        "bm25": ENSEMBLE_WEIGHTS[0],
        "faiss": ENSEMBLE_WEIGHTS[1],
    }

    logger.info(
        f"[RAG] [RETRIEVE] 하이브리드 가중치 적용: BM25({weights['bm25']:.1f}), FAISS({weights['faiss']:.1f})"
    )

    aggregated, _ = aggregator.aggregate_results(
        {"all": all_docs},
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=25,
        weights=weights,
    )

    # [최적화] Dynamic Top-K 결정: 상위권 점수 격차(Gap)가 크면 리랭킹 후보군 축소
    if len(aggregated) >= 10:
        top_1_score = aggregated[0].aggregated_score
        top_10_score = aggregated[9].aggregated_score
        score_gap = top_1_score - top_10_score

        # Gap이 0.5 이상이면 상위 그룹이 명확하므로 후보군을 12개로 제한
        dynamic_top_k = 12 if score_gap > 0.5 else 18
        logger.info(
            f"[RAG] [RETRIEVE] Dynamic Top-K 적용: {dynamic_top_k} (Score Gap: {score_gap:.3f})"
        )
    else:
        dynamic_top_k = 12

    final_docs = [
        doc_map.get(
            r.doc_id,
            Document(page_content=r.content, metadata=r.metadata),
        )
        for r in aggregated[:dynamic_top_k]
    ]

    if not final_docs:
        q_len = len(query) if query else 0
        logger.warning(
            f"[RAG] [RETRIEVE] 검색 결과가 전혀 없습니다 (Query Length: {q_len})"
        )
        SessionManager.add_status_log(
            "검색된 문서가 없습니다.", session_id=_get_session_id(config)
        )
        return {"relevant_docs": []}

    from core.async_reranker import get_async_reranker

    reranker = await get_async_reranker()
    rerank_top_k = min(GRADING_CONFIG.get("top_k", 5), len(final_docs))
    if len(query or "") < 6:
        ranked_docs = final_docs[:rerank_top_k]
    else:
        ranked_docs, _ = await reranker.rerank(
            final_docs,
            query=query,
            top_k=rerank_top_k,
        )
    logger.info(
        f"[RAG] [RETRIEVE] 리랭킹 선별 완료: {len(final_docs)}개 후보 중 {len(ranked_docs)}개 최종 선별"
    )

    from typing import cast

    context_docs = cast(list[Document], ranked_docs)
    merged_context_docs = await asyncio.to_thread(
        _merge_adjacent_chunks, context_docs, max_tokens=800
    )
    logger.info(
        f"[RAG] [RETRIEVE] 하이브리드 검색 및 문맥 보강 완료: 최종 {len(merged_context_docs)}개 섹션 구성"
    )

    return {"relevant_docs": merged_context_docs}


class UnifiedGradeRewriteResponse(BaseModel):
    """문서 평가 + 쿼리 재구성을 단일 호출로 통합."""

    action: str = Field(description="'generate' 또는 'rewrite'")
    is_relevant: bool = Field(
        description="문서가 질문에 답변하기에 충분한 정보를 포함하고 있는지 여부"
    )
    relevant_entities: list[str] = Field(
        default_factory=list,
        description="질문과 관련된 문서 내 핵심 키워드나 고유 명사 목록",
    )
    reason: str = Field(description="결정에 대한 구체적인 근거")
    optimized_query: str | None = Field(
        default=None,
        description="rewrite 시에만 채울 최적화된 검색어",
    )


async def grade_documents(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """검색된 문서들의 관련성을 LLM으로 평가하고, 부적절 시 재검색 쿼리를 동시에 생성합니다."""
    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        return {}

    retry_count = get_state_attr(state, "retry_count", 0)
    if retry_count >= 2:
        logger.info(
            f"[RAG] [GRADE] 최대 재시도 횟수({retry_count}) 도달. 즉시 생성 단계로 이동."
        )
        return {"intent": "generate"}

    docs = get_state_attr(state, "relevant_docs")
    if not docs:
        logger.info("[RAG] [GRADE] 문서가 없어 즉시 재구성 단계로 이동")
        return {"intent": "transform"}

    # Short-circuit: 리랭킹 점수가 충분히 높으면 LLM 검증 생략
    max_rerank_score = max(
        (d.metadata.get("rerank_score", 0.0) for d in docs), default=0.0
    )
    if max_rerank_score >= 0.85:
        logger.info(
            f"[RAG] [GRADE] Short-circuit 활성화 (Max Rerank Score: {max_rerank_score:.3f} >= 0.85)"
        )
        SessionManager.add_status_log(
            "신뢰도 높은 지식이 발견되어 즉시 답변 생성을 시작합니다.",
            session_id=_get_session_id(config),
        )
        return {"intent": "generate"}

    query = get_state_attr(state, "input")
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import json
    import re

    if writer is not None:
        await adispatch_custom_event(
            "graph_status", {"status": "문서 관련성 검증 중..."}, config=config
        )

    test_docs = docs[:3]
    context_text = "\n\n".join(
        [f"DOC {i + 1}: {d.page_content}" for i, d in enumerate(test_docs)]
    )

    # 통합 프롬프트: 평가 + 재작성 동시 지시
    unified_prompt = (
        "당신은 문서 관련성 평가자이자 검색 쿼리 최적화 전문가입니다.\n\n"
        f"[질문]\n{query}\n\n"
        f"[검색된 문서 (상위 3개)]\n{context_text}\n\n"
        "[작업]\n"
        "1. 위 문서들이 질문에 답하기에 충분한지 평가하세요 (is_relevant: true/false)\n"
        "2. 충분하지 않다면, 더 나은 검색 결과를 위한 최적화된 쿼리를 작성하세요 (optimized_query)\n"
        "3. 판단 근거(reason)와 관련 엔티티(relevant_entities)도 포함하세요.\n\n"
        '출력은 반드시 JSON 형식이어야 합니다. (예: {"action": "generate", "is_relevant": true, '
        '"relevant_entities": ["A"], "reason": "...", "optimized_query": null})'
    )

    call_config = (
        {"configurable": {**cfg, "messages": []}}
        if cfg
        else {"configurable": {"messages": []}}
    )

    try:
        if llm is None:
            raise ValueError("LLM is not initialized")

        # JSON 모드 강제 (구조화 출력 대신) — 단일 호출로 완성
        try:
            async with ModelManager.inference_session():
                json_llm = llm.bind(response_format={"type": "json_object"})
                result = await _safe_invoke(json_llm, unified_prompt, call_config)
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)
            parsed = UnifiedGradeRewriteResponse(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[RAG] [GRADE] JSON 모드 실패, 수동 파싱 시도: {e}")
            async with ModelManager.inference_session():
                raw_res = await _safe_invoke(llm, unified_prompt, call_config)
            raw_content = (
                raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            )
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                parsed = UnifiedGradeRewriteResponse(**data)
            else:
                raise ValueError("JSON 패턴을 찾을 수 없습니다.") from None

        if parsed.action == "generate" or parsed.is_relevant:
            logger.info(f"[RAG] [GRADE] 관련성 확인: YES ({parsed.reason})")
            SessionManager.add_status_log(
                "검색된 지식의 관련성이 확인되었습니다.",
                session_id=_get_session_id(config),
            )
            return {"intent": "generate"}
        else:
            optimized = parsed.optimized_query or query
            logger.info(
                f"[RAG] [GRADE] 관련성 확인: NO → 재작성: {optimized} ({parsed.reason})"
            )
            SessionManager.add_status_log(
                "검색 결과가 부적합하여 질문 재구성을 시도합니다.",
                session_id=_get_session_id(config),
            )
            return {
                "intent": "transform",
                "search_queries": [optimized],
                "retry_count": retry_count + 1,
            }

    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[RAG] [GRADE] 평가 실패, 기본값(NO) 적용하여 재구성 시도: {e}")
        return {"intent": "transform", "retry_count": retry_count + 1}


async def rewrite_query(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """grade_documents에서 이미 재작성된 쿼리를 전달합니다. (LLM 호출 없음)"""
    MAX_REWRITE_RETRIES = 3
    search_queries = get_state_attr(state, "search_queries")
    retry_count = get_state_attr(state, "retry_count", 0)

    if search_queries:
        new_query = search_queries[-1]
        logger.info(f"[RAG] [REWRITE] 전달받은 재구성 쿼리 사용: '{new_query}'")
        # retry_count는 grade_documents가 이미 +1을 적용했으므로 순수 passthrough만 한다.
        # (리듀서 operator.add로 합산되므로 여기서 값을 내려보내면 재시도 예산이 이중 소진됨)
        return {}

    # 폴백: grade_documents에서 쿼리 생성 실패 시 원본 유지
    query = get_state_attr(state, "input")
    logger.info(f"[RAG] [REWRITE] 재검색 쿼리 없음, 원본 유지: '{query}'")
    return {"retry_count": min(retry_count + 1, MAX_REWRITE_RETRIES)}


def format_context(docs: list[Document]) -> str:
    """검색된 문서들을 LLM이 읽기 좋은 형식의 문자열로 변환합니다."""
    context = ""
    for _i, d in enumerate(docs):
        section = d.metadata.get("current_section", "일반 본문")
        page = d.metadata.get("page", "?")
        context += f"[{section}, p.{page}]\n{d.page_content}\n\n"
    return context


async def generate(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """최종 답변을 생성합니다."""
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"response": "LLM 미로드"}

    if writer is not None:
        await adispatch_custom_event(
            "graph_status", {"status": "답변 논리 설계 및 생성 중..."}, config=config
        )
    SessionManager.add_status_log(
        "답변 논리 설계 및 생성 시작", session_id=_get_session_id(config)
    )

    docs = get_state_attr(state, "relevant_docs") or []
    logger.info(f"[RAG] [GENERATE] 관련 문서 수: {len(docs) if docs else 0}")
    if docs:
        for i, d in enumerate(docs):
            logger.info(f"[RAG] [GENERATE] 문서 {i} 길이: {len(d.page_content)}")
    if not docs and get_state_attr(state, "intent") != "general":
        logger.info("[RAG] [GENERATE] 관련 문서 없음 -> 사용자 안내 메시지 생성")
        no_info_msg = "제공된 문서에서 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 입력하거나 문서 내용을 확인해 주세요."
        if writer is not None:
            await adispatch_custom_event(
                "response_chunk", {"content": no_info_msg}, config=config
            )
        return {"response": no_info_msg}

    context = format_context(docs) if docs else "일상적인 대화입니다."

    sys_msg = SystemMessage(
        content="전문 문서 분석가입니다. 사용자의 질문 언어에 맞추어 답변하십시오."
    )
    human_msg = HumanMessage(
        content=f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{get_state_attr(state, 'input')}"
    )

    full_response = ""
    full_thought = ""
    last_metadata = {}

    async with ModelManager.inference_session():
        async for chunk in llm.astream([sys_msg, human_msg], config=config):
            if hasattr(llm, "_convert_chunk_to_thought_and_content"):
                content_chunk, thought_chunk = (
                    llm._convert_chunk_to_thought_and_content(chunk)
                )
            else:
                # Fallback for LLMs without thought/content separation
                content_chunk = (
                    chunk.content if hasattr(chunk, "content") else str(chunk)
                )
                thought_chunk = ""
            if (content_chunk or thought_chunk) and writer is not None:
                await adispatch_custom_event(
                    "response_chunk",
                    {"content": content_chunk, "thought": thought_chunk},
                    config=config,
                )

            if thought_chunk:
                full_thought += thought_chunk
            if content_chunk:
                full_response += content_chunk
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                last_metadata = chunk.response_metadata

    input_tokens = last_metadata.get("prompt_eval_count", 0)
    return {
        "response": full_response,
        "thought": full_thought,
        "performance": {
            **last_metadata,
            "input_token_count": input_tokens,
            "relevant_docs_count": len(docs),
        },
    }


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
        else:
            merged_docs.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content,
                metadata=copy.copy(next_doc.metadata),
            )
            current_tokens = next_tokens

    merged_docs.append(current_doc)
    return merged_docs


async def build_graph() -> Any:
    """자가 교정형 RAG 워크플로우를 구성합니다.

    [최적화] asyncio.Lock을 사용하여 동시 빌드 요청 시 이중 컴파일을 방지합니다.
    """
    logger.info("[RAG] [GRAPH] build_graph() 호출됨")
    if _graph_cache.compiled is not None:
        logger.info("[RAG] [GRAPH] 캐시된 그래프 반환")
        return _graph_cache.compiled

    lock = _graph_cache.get_lock()
    async with lock:
        # Double-check after acquiring lock (다른 세션이 이미 빌드 완료했을 수 있음)
        if _graph_cache.compiled is not None:
            logger.info("[RAG] [GRAPH] 락 획득 후 캐시된 그래프 반환")
            return _graph_cache.compiled

        logger.info("[RAG] [GRAPH] 그래프 재구성 시작")
        workflow = StateGraph(GraphState)

        # 노드 등록
        workflow.add_node("preprocess", preprocess)
        workflow.add_node("retrieve", retrieve_and_rerank)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("generate", generate)

        # 엣지 설정
        workflow.add_edge(START, "preprocess")

        workflow.add_conditional_edges(
            "preprocess",
            lambda s: (
                "generate" if get_state_attr(s, "intent") == "general" else "retrieve"
            ),
            {"generate": "generate", "retrieve": "retrieve"},
        )

        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            lambda s: get_state_attr(
                s, "intent", "generate"
            ),  # default to generate on unknown
            {"generate": "generate", "transform": "rewrite_query"},
        )

        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", END)

        # LangChain pending deprecation warning (allowed_objects 기본값 변경 예정) 침묵화
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*allowed_objects.*",
        )
        from langgraph.checkpoint.memory import InMemorySaver

        memory = InMemorySaver()

        _graph_cache.compiled = workflow.compile(checkpointer=memory)
        return _graph_cache.compiled
