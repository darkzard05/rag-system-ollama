"""
LangGraph를 사용하여 단순화된 RAG 워크플로우를 구성합니다.
의도 분류, 캐시 확인, 하이브리드 검색, 생성의 핵심 단계를 직선화합니다.
"""

import asyncio
import logging
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from api.schemas import AggregatedSearchResult, GraphState
from cache.response_cache import get_response_cache
from common.config import (
    ANALYSIS_PROTOCOL,
    RERANKER_CONFIG,
)
from common.utils import fast_hash
from core.session import SessionManager
from services.monitoring.llm_tracker import ResponsePerformanceTracker

logger = logging.getLogger(__name__)


async def preprocess(state: GraphState) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 동시에 수행합니다."""
    query = state["input"].strip()

    # 1. 의도 분류 (단순 규칙)
    greetings = ["안녕", "hi", "hello", "도움말", "고마워"]
    if any(g in query.lower() for g in greetings) or len(query) < 2:
        return {"intent": "general", "is_cached": False}

    # 2. 캐시 확인
    cache = get_response_cache()
    cached_res = await cache.get(query, use_semantic=True)
    if cached_res:
        SessionManager.add_status_log("캐시된 답변을 발견했습니다.")
        return {
            "response": cached_res.response,
            "thought": cached_res.metadata.get("thought", ""),
            "is_cached": True,
        }

    return {"intent": "rag", "is_cached": False}


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if state.get("is_cached") or state.get("intent") == "general":
        return {}

    from common.config import RAG_PARAMETERS
    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    query = state["input"]
    cfg = config.get("configurable", {})
    RAG_PARAMETERS.get("retrieval_k", 25)

    # [핵심] UI 핸들러를 위한 상태 이벤트 발생
    await adispatch_custom_event(
        "status_update", {"message": "🔍 관련 지식 탐색 중..."}, config=config
    )
    SessionManager.add_status_log("🔍 문서 저장소에서 관련 지식 탐색 시작")

    # 리트리버 획득
    bm25 = cfg.get("bm25_retriever")
    faiss = cfg.get("faiss_retriever")

    # 병렬 검색
    tasks = []
    if bm25:
        tasks.append(asyncio.create_task(bm25.ainvoke(query)))
    if faiss:
        tasks.append(asyncio.create_task(faiss.ainvoke(query)))

    results = await asyncio.gather(*tasks) if tasks else [[], []]

    # 결과 병합 및 RRF 집계
    all_docs = []
    for i, res in enumerate(results):
        source = "bm25" if i == 0 else "faiss"
        for doc in res:
            all_docs.append(
                AggregatedSearchResult(
                    doc_id=doc.metadata.get("doc_id", fast_hash(doc.page_content)),
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.5),
                    node_id=source,
                    metadata=doc.metadata,
                )
            )

    aggregator = SearchResultAggregator()
    weights = {"bm25": 0.4, "faiss": 0.6}
    aggregated, _ = aggregator.aggregate_results(
        {"all": all_docs},
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=25,
        weights=weights,
    )

    final_docs = [
        Document(page_content=r.content, metadata=r.metadata) for r in aggregated
    ]
    SessionManager.add_status_log(
        f"📚 하이브리드 검색 완료 ({len(final_docs)}개 후보 확보)"
    )

    # FlashRank 리랭킹 (선택적)
    if RERANKER_CONFIG.get("enabled", True) and len(final_docs) > 1:
        await adispatch_custom_event(
            "status_update",
            {"message": "⚖️ 지식 우선순위 정제 중 (FlashRank)"},
            config=config,
        )
        SessionManager.add_status_log("⚖️ 지식의 우선순위 재조정 및 정제 중 (FlashRank)")
        from flashrank import RerankRequest

        from core.model_loader import ModelManager

        ranker = await ModelManager.get_flashranker()
        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(final_docs)
        ]

        await ModelManager.acquire_inference_lock()
        try:
            results = await asyncio.to_thread(
                ranker.rerank, RerankRequest(query=query, passages=passages)
            )
            final_docs = [
                Document(page_content=r["text"], metadata=r["meta"])
                for r in results[:10]
            ]
            SessionManager.add_status_log(
                f"✅ 최적의 지식 10개 선별 완료 (신뢰도: {results[0]['score']:.2f})"
            )
        finally:
            ModelManager.release_inference_lock()

    return {"relevant_docs": final_docs}


async def generate(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """최종 답변을 생성합니다."""
    if state.get("is_cached"):
        return {}

    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"response": "❌ LLM 미로드"}

    await adispatch_custom_event(
        "status_update", {"message": "✍️ 답변 작성 중..."}, config=config
    )
    SessionManager.add_status_log("🧠 답변 논리 설계 및 생성 시작")

    # 컨텍스트 포맷팅
    docs = state.get("relevant_docs", [])
    context = ""
    for i, d in enumerate(docs):
        context += (
            f"### [자료 {i + 1}] (P{d.metadata.get('page', '?')})\n{d.page_content}\n\n"
        )

    # 프롬프트 구성
    sys_msg = SystemMessage(content="전문 문서 분석가로서 한국어로 답변하세요.")
    human_msg = HumanMessage(
        content=f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{state['input']}"
    )

    tracker = ResponsePerformanceTracker(state["input"], llm)
    tracker.set_context(context, doc_count=len(docs))

    from core.model_loader import ModelManager

    await ModelManager.acquire_inference_lock()
    try:
        async for chunk in llm.astream([sys_msg, human_msg], config=config):
            msg = getattr(chunk, "message", chunk)
            content, thought = msg.content, msg.additional_kwargs.get("thinking", "")
            tracker.record_chunk(content, thought)
            if content or thought:
                await adispatch_custom_event(
                    "response_chunk",
                    {"chunk": content, "thought": thought},
                    config=config,
                )
    finally:
        ModelManager.release_inference_lock()

    # 성능 지표 확정 및 반환 데이터 구성
    stats = tracker.finalize_and_log()
    return {
        "response": tracker.full_response,
        "thought": tracker.full_thought,
        "performance": stats.model_dump() if hasattr(stats, "model_dump") else stats,
    }


def build_graph() -> Any:
    """그래프를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("retrieve", retrieve_and_rerank)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "preprocess")

    # 조건부 에지: 캐시가 있으면 바로 종료, 일상 대화면 바로 생성, 아니면 검색
    workflow.add_conditional_edges(
        "preprocess",
        lambda s: (
            "END"
            if s.get("is_cached")
            else ("generate" if s.get("intent") == "general" else "retrieve")
        ),
        {"END": END, "generate": "generate", "retrieve": "retrieve"},
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
