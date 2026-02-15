"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Strict Linear Pipeline: 의도 분류 없이 모든 입력을 동일한 고성능 경로로 처리합니다.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from api.schemas import AggregatedSearchResult, GraphState
from cache.response_cache import get_response_cache
from common.config import (
    ANALYSIS_PROTOCOL,
    QUERY_EXPANSION_CONFIG,
    QUERY_EXPANSION_PROMPT,
    RERANKER_CONFIG,
)
from common.typing_utils import (
    DocumentList,
    GraphOutput,
)
from common.utils import fast_hash
from core.session import SessionManager
from services.monitoring.llm_tracker import ResponsePerformanceTracker
from services.monitoring.performance_monitor import (
    get_performance_monitor,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


class QueryExpansionOutput(BaseModel):
    """쿼리 확장 결과 스키마"""

    queries: list[str] = Field(description="확장된 검색 쿼리 리스트 (최대 3개)")


# --- Helper Functions ---


def _merge_consecutive_chunks(docs: DocumentList) -> DocumentList:
    """
    검색된 청크들 사이의 겹치는 내용을 제거하고, 동일 섹션/페이지 내의 내용을 지능적으로 병합합니다.
    """
    if not docs:
        return []

    # 1. 문서 순서 유지하며 병합
    merged: DocumentList = []
    current_doc = Document(
        page_content=docs[0].page_content, metadata=docs[0].metadata.copy()
    )

    for i in range(1, len(docs)):
        next_doc = docs[i]

        # [최적화] 동일 섹션(last_section)이거나 동일 페이지인 경우 병합 우선순위 부여
        is_same_section = (
            next_doc.metadata.get("last_section")
            == current_doc.metadata.get("last_section")
            and next_doc.metadata.get("last_section") != "General"
        )
        is_same_page = next_doc.metadata.get("source") == current_doc.metadata.get(
            "source"
        ) and next_doc.metadata.get("page") == current_doc.metadata.get("page")

        if is_same_section or is_same_page:
            text_a = current_doc.page_content
            text_b = next_doc.page_content

            # 오버랩 제거 로직
            overlap_len = 0
            max_check = min(len(text_a), len(text_b), 350)

            for length in range(max_check, 5, -1):
                if text_a.endswith(text_b[:length]):
                    overlap_len = length
                    break

            if overlap_len > 0:
                current_doc.page_content += text_b[overlap_len:]
            else:
                # 섹션이 같으면 줄바꿈으로 연결, 다르면 구분자 사용
                separator = "\n" if is_same_section else "\n[...]\n"
                current_doc.page_content += separator + text_b
        else:
            merged.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content, metadata=next_doc.metadata.copy()
            )

    merged.append(current_doc)
    return merged


# --- Graph Construction ---


def build_graph(retriever: Any = None) -> Any:
    """
    RAG 워크플로우를 구성하고 컴파일합니다.
    """

    # 0. 의도 분류 노드 (최적화: 검색 필요 여부 판단)
    async def classify_intent(state: GraphState, config: RunnableConfig) -> GraphOutput:
        query = state["input"].strip()

        # 가벼운 규칙 기반 분류 (속도 우선)
        greetings = [
            "안녕",
            "hi",
            "hello",
            "반가워",
            "누구야",
            "도움말",
            "고마워",
            "thanks",
        ]
        if any(g in query.lower() for g in greetings) or len(query) < 2:
            logger.info("[Router] 일상 대화 또는 짧은 쿼리 감지 -> 검색 건너뜀")
            return {"intent": "general", "is_cached": False}

        return {"intent": "rag"}

    # 0-1. 캐시 확인 노드
    async def check_cache(state: GraphState) -> GraphOutput:
        if state.get("intent") == "general":
            return {"is_cached": False}

        cache = get_response_cache()
        cached_res = await cache.get(state["input"], use_semantic=True)

        if cached_res:
            logger.info(f"[Cache] 히트! 캐시된 응답 사용: {state['input'][:50]}")
            SessionManager.replace_last_status_log("캐시된 답변을 발견했습니다.")
            return {
                "response": cached_res.response,
                "thought": cached_res.metadata.get("thought", ""),
                "performance": cached_res.metadata.get("performance", {}),
                "is_cached": True,
            }
        return {"is_cached": False}

    # 2. 문서 검색 및 확장 노드 (병렬 최적화)
    async def retrieve_documents(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        from common.config import RAG_PARAMETERS
        from core.search_aggregator import AggregationStrategy, SearchResultAggregator

        query = state["input"].strip()
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        k_val = RAG_PARAMETERS.get("retrieval_k", 25)

        bm25 = configurable.get("bm25_retriever") or SessionManager.get(
            "bm25_retriever", session_id=session_id
        )
        faiss_ret = configurable.get("faiss_retriever") or SessionManager.get(
            "faiss_retriever", session_id=session_id
        )

        async def _invoke_retriever(ret: Any, q: str, k: int) -> DocumentList:
            if not ret:
                return []
            try:
                if hasattr(ret, "search_kwargs"):
                    ret.search_kwargs["k"] = k
                if hasattr(ret, "ainvoke"):
                    return await asyncio.wait_for(ret.ainvoke(q), timeout=15.0)
                return await asyncio.to_thread(ret.invoke, q)
            except Exception:
                return []

        # [최적화] 1. 원본 쿼리 검색 즉시 시작 (백그라운드 태스크)
        initial_tasks = [
            ("bm25_0", _invoke_retriever(bm25, query, k_val)),
            ("faiss_0", _invoke_retriever(faiss_ret, query, k_val)),
        ]

        # [최적화] 2. 동시에 쿼리 확장 수행 (인텐트가 RAG인 경우에만)
        async def _expand_query():
            if state.get("intent") == "general" or not QUERY_EXPANSION_CONFIG.get(
                "enabled", True
            ):
                return []
            llm = configurable.get("llm") or SessionManager.get("llm")
            if not llm:
                return []
            try:
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(QueryExpansionOutput)
                    prompt = ChatPromptTemplate.from_messages(
                        [("system", QUERY_EXPANSION_PROMPT), ("human", "{input}")]
                    )
                    res = await (prompt | structured_llm).ainvoke(
                        {"input": query}, config=config
                    )
                    return [q.strip() for q in res.queries if len(q.strip()) > 2][
                        :2
                    ]  # 최대 2개만 추가
                return []
            except Exception:
                return []

        # 3. 병렬 실행: (원본 검색) + (쿼리 확장)
        expansion_task = asyncio.create_task(_expand_query())
        initial_results_list = await asyncio.gather(*[t[1] for t in initial_tasks])
        expanded_queries = await expansion_task

        # 4. 확장 쿼리에 대한 추가 검색 수행
        secondary_tasks = []
        for i, q in enumerate(expanded_queries):
            secondary_tasks.append((f"bm25_ext_{i}", _invoke_retriever(bm25, q, k_val)))
            secondary_tasks.append(
                (f"faiss_ext_{i}", _invoke_retriever(faiss_ret, q, k_val))
            )

        secondary_results_list = (
            await asyncio.gather(*[t[1] for t in secondary_tasks])
            if secondary_tasks
            else []
        )

        # 5. 모든 결과 수집 및 집계
        search_results_map = {}
        for i, res in enumerate(initial_results_list):
            if res:
                search_results_map[initial_tasks[i][0]] = res
        for i, res in enumerate(secondary_results_list):
            if res:
                search_results_map[secondary_tasks[i][0]] = res

        wrapped_results = {
            node_id: [
                AggregatedSearchResult(
                    doc_id=doc.metadata.get("doc_id", fast_hash(doc.page_content)),
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.5),
                    node_id=node_id.split("_")[0],
                    metadata=doc.metadata,
                )
                for doc in docs
            ]
            for node_id, docs in search_results_map.items()
        }

        aggregator = SearchResultAggregator()
        from common.config import RETRIEVER_CONFIG

        weights = {
            "bm25": RETRIEVER_CONFIG.get("ensemble_weights", [0.4, 0.6])[0],
            "faiss": RETRIEVER_CONFIG.get("ensemble_weights", [0.4, 0.6])[1],
        }

        aggregated, _ = aggregator.aggregate_results(
            wrapped_results,
            strategy=AggregationStrategy.WEIGHTED_RRF,
            top_k=30,
            weights=weights,
        )

        final_docs = [
            Document(page_content=r.content, metadata=r.metadata) for r in aggregated
        ]
        logger.info(f"[Retrieve] {len(final_docs)}개 문서 수집 (병렬 검색 완료)")
        return {"documents": final_docs}

    # 3. 재순위화 노드 (FlashRank 고속화 적용)
    async def rerank_documents(state: GraphState) -> GraphOutput:
        from core.model_loader import ModelManager

        documents = state.get("documents", [])
        if not RERANKER_CONFIG.get("enabled", True) or len(documents) <= 1:
            return {"documents": documents}

        semaphore = ModelManager.get_inference_semaphore()

        try:
            # 1. FlashRank 모델 로드 (캐시됨)
            model_name = RERANKER_CONFIG.get("model_name", "ms-marco-MiniLM-L-12-v2")
            ranker = ModelManager.get_flashranker(model_name)

            # 2. FlashRank 포맷 변환
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(documents)
            ]

            # 3. 리랭킹 수행 (CPU 스레드) - 세마포어 적용
            from flashrank import RerankRequest

            rank_request = RerankRequest(query=state["input"], passages=passages)

            async with semaphore:
                results = await asyncio.to_thread(ranker.rerank, rank_request)

            # 4. 필터링 및 복원
            final_docs: list[Document] = []
            min_score = RERANKER_CONFIG.get("min_score", 0.1)
            top_k = RERANKER_CONFIG.get("top_k", 10)  # [최적화] 컨텍스트 풍부화

            for res in results:
                if res["score"] >= min_score or len(final_docs) < 3:
                    final_docs.append(
                        Document(page_content=res["text"], metadata=res["meta"])
                    )
                if len(final_docs) >= top_k:
                    break

            logger.info(
                f"[FlashRank] {len(documents)} -> {len(final_docs)}개 선별 (최고점: {results[0]['score']:.3f})"
            )
            return {"documents": final_docs}
        except Exception as e:
            logger.error(f"[FlashRank] 오류: {e}")
            return {"documents": documents[:10]}

    # 4. 문서 채점 노드 (통과)
    async def grade_documents(state: GraphState) -> GraphOutput:
        return {"relevant_docs": state.get("documents", [])}

    # 5. 컨텍스트 포맷팅 노드
    async def format_context(state: GraphState) -> GraphOutput:
        documents = state.get("relevant_docs", [])
        if not documents:
            return {"context": ""}

        merged_docs = _merge_consecutive_chunks(documents)
        formatted = []
        for i, doc in enumerate(merged_docs):
            page = doc.metadata.get("page", "?")
            formatted.append(f"### [자료 {i + 1}] (P{page})\n{doc.page_content}")

        return {"context": "\n\n".join(formatted)}

    # 6. 답변 생성 노드
    async def generate_response(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        configurable = config.get("configurable", {})
        sid = configurable.get("session_id")
        llm = configurable.get("llm") or SessionManager.get("llm", session_id=sid)

        try:
            context = state.get("context", "").strip()
            user_input = state.get("input", "").strip()

            # [최적화] ANALYSIS_PROTOCOL의 가이드라인과 중복되는 내용 제거 및 명확화
            system_instruction = (
                f"{ANALYSIS_PROTOCOL}\n\n"
                "[Special Instructions]\n"
                "- 인사말인 경우 전문가로서 짧게 화답하십시오.\n"
                "- 분석 요청인 경우 반드시 한국어로 답변하며, <Context> 내의 구체적 근거(수치, 용어)를 활용하십시오.\n"
                "- 가독성을 위해 불렛 포인트와 구조화된 형식을 사용하십시오."
            )

            final_human_prompt = (
                f"[지시사항]\n{system_instruction}\n\n"
                f"[참고 문헌 컨텍스트]\n{context or '(제공된 자료 없음)'}\n\n"
                f"[사용자 질문]\n{user_input}"
            )

            messages = [
                SystemMessage(
                    content="You are a professional document analysis expert. Always respond in Korean."
                ),
                HumanMessage(content=final_human_prompt),
            ]

            logger.info(f"[Response-Node] 생성 시작 (Context: {len(context)} chars)")

            tracker = ResponsePerformanceTracker(user_input, llm)
            tracker.set_context(context)

            from core.model_loader import ModelManager

            semaphore = ModelManager.get_inference_semaphore()

            async with semaphore:
                async for chunk in llm.astream(messages, config=config):
                    msg = getattr(chunk, "message", chunk)
                    content = msg.content
                    thinking = msg.additional_kwargs.get("thinking", "")
                    tracker.record_chunk(content, thinking)
                    if content or thinking:
                        await adispatch_custom_event(
                            "response_chunk",
                            {"chunk": content, "thought": thinking},
                            config=config,
                        )

            stats = tracker.finalize_and_log()
            return {
                "response": tracker.full_response,
                "thought": tracker.full_thought,
                "performance": stats.model_dump(),
            }
        except Exception as e:
            logger.error(f"[Generate] 오류: {e}")
            raise

    # Workflow Definition
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        lambda s: "END"
        if s.get("is_cached")
        else ("generate_response" if s.get("intent") == "general" else "retrieve"),
        {"END": END, "generate_response": "generate_response", "retrieve": "retrieve"},
    )
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_documents")
    workflow.add_edge("grade_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
