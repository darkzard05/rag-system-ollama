"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Core Logic Rebuild: 데코레이터 제거 및 순수 함수 구조로 변경하여 config 전달 보장.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from api.schemas import AggregatedSearchResult, GraphState
from cache.response_cache import get_response_cache
from common.config import (
    ANALYSIS_PROTOCOL,
    FACTOID_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    INTENT_PARAMETERS,
    OUT_OF_CONTEXT_SYSTEM_PROMPT,
    QA_HUMAN_PROMPT,
    QUERY_EXPANSION_CONFIG,
    QUERY_EXPANSION_PROMPT,
    RERANKER_CONFIG,
    RESEARCH_SYSTEM_PROMPT,
)
from common.typing_utils import (
    DocumentList,
    GraphOutput,
)
from common.utils import count_tokens_rough, fast_hash
from core.model_loader import load_reranker_model
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
    같은 출처의 연속된 인덱스를 가진 청크들을 하나로 병합합니다.
    """
    if not docs:
        return []

    sorted_docs = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0),
            d.metadata.get("chunk_index", -1),
        ),
    )

    merged: DocumentList = []
    if not sorted_docs:
        return merged

    current_doc = Document(
        page_content=sorted_docs[0].page_content,
        metadata=sorted_docs[0].metadata.copy(),
    )

    for next_doc in sorted_docs[1:]:
        curr_src = current_doc.metadata.get("source")
        next_src = next_doc.metadata.get("source")
        curr_page = current_doc.metadata.get("page")
        next_page = next_doc.metadata.get("page")

        curr_idx = current_doc.metadata.get("chunk_index", -1)
        next_idx = next_doc.metadata.get("chunk_index", -1)

        if (
            curr_src == next_src
            and curr_page == next_page
            and curr_idx != -1
            and next_idx == curr_idx + 1
        ):
            current_doc.page_content += " " + next_doc.page_content
            current_doc.metadata["chunk_index"] = next_idx
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

    # 0-1. 캐시 확인 노드
    async def check_cache(state: GraphState) -> GraphOutput:
        cache = get_response_cache()
        cached_res = await cache.get(state["input"], use_semantic=True)

        if cached_res:
            logger.info(f"[Cache] 히트! 캐시된 응답 사용: {state['input'][:50]}")
            SessionManager.replace_last_status_log(
                "유사한 질문에 대한 캐시된 답변을 발견했습니다."
            )
            return {
                "response": cached_res.response,
                "thought": cached_res.metadata.get("thought", ""),
                "route_decision": cached_res.metadata.get("route_decision", "CACHE"),
                "performance": cached_res.metadata.get("performance", {}),
                "is_cached": True,
            }
        return {"is_cached": False}

    # 0-2. 의도 분석 노드
    async def router(state: GraphState, config: RunnableConfig) -> GraphOutput:
        from common.config import INTENT_ANALYSIS_ENABLED
        from core.query_optimizer import RAGQueryOptimizer

        # Config에서 LLM 추출
        configurable = config.get("configurable", {})
        llm = configurable.get("llm") or SessionManager.get("llm")

        # [최적화] 의도 분석 비활성화 시 기본값 반환
        if not INTENT_ANALYSIS_ENABLED:
            logger.info(
                "[CHAT] [ROUTER] 의도 분석 비활성화됨 -> 기본값 'RESEARCH' 사용"
            )
            return {"route_decision": "RESEARCH"}

        if not llm:
            return {"route_decision": "FACTOID"}

        intent = await RAGQueryOptimizer.classify_intent(state["input"], llm)
        SessionManager.replace_last_status_log(
            f"의도 분석 완료: {intent}", session_id=configurable.get("session_id")
        )
        return {"route_decision": intent}

    # 1. 쿼리 확장 노드
    async def generate_queries(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        route = state.get("route_decision", "FACTOID")
        configurable = config.get("configurable", {})

        # [최적화] 쿼리 확장 비활성화 시 즉시 반환
        if not QUERY_EXPANSION_CONFIG.get("enabled", True) or route == "GREETING":
            return {"search_queries": [state["input"]]}

        llm = configurable.get("llm") or SessionManager.get("llm")
        if not llm:
            return {"search_queries": [state["input"]]}

        # [고도화] 의도에 따른 검색 전략 차별화
        if route == "SUMMARY":
            instruction = (
                f"사용자가 다음 문서에 대한 전체 요약 또는 재구성을 요청했습니다: '{state['input']}'\n"
                "문서의 핵심 골격을 파악하기 위해 다음 3가지 관점의 고유명사를 포함한 검색어를 생성하십시오:\n"
                "1. 해당 주제의 서론(Introduction) 및 초록(Abstract)\n"
                "2. 해당 주제의 주요 결론(Conclusion) 및 핵심 결과\n"
                "3. 이 문서만의 독창적인 기여도(Contributions) 및 핵심 아키텍처"
            )
        else:
            doc_lang = (
                configurable.get("doc_language")
                or SessionManager.get("doc_language")
                or "English"
            )
            instruction = f"{QUERY_EXPANSION_PROMPT}"
            if doc_lang == "English":
                instruction += "\n주의: 대상 문서가 영어이므로 검색 효율을 위해 확장 쿼리 중 2개 이상은 반드시 영어로 작성하십시오."

        try:
            if hasattr(llm, "with_structured_output"):
                structured_llm = llm.with_structured_output(QueryExpansionOutput)
                prompt = ChatPromptTemplate.from_messages(
                    [("system", instruction), ("human", "{input}")]
                )
                chain = prompt | structured_llm
                result = await chain.ainvoke({"input": state["input"]}, config=config)
                raw_queries = result.queries
            else:
                raw_queries = [state["input"]]

            clean_queries = [state["input"]]
            for q in raw_queries:
                q = q.strip().replace('"', "").replace("'", "")
                if q and len(q) > 2 and q not in clean_queries:
                    clean_queries.append(q)

            final_queries = clean_queries[:4]
            logger.info(f"[Query] 의도({route}) 기반 확장 완료: {final_queries}")
            SessionManager.replace_last_status_log(
                f"의도({route}) 기반 검색 전략 수립 완료",
                session_id=configurable.get("session_id"),
            )
            return {"search_queries": final_queries}
        except Exception as e:
            logger.error(f"[Query] 확장 오류: {e}")
            return {"search_queries": [state["input"]]}

    # 2. 문서 검색 노드
    async def retrieve_documents(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        from core.search_aggregator import AggregationStrategy, SearchResultAggregator

        route = state.get("route_decision", "FACTOID")
        params = INTENT_PARAMETERS.get(route, INTENT_PARAMETERS.get("DEFAULT"))
        k_val = params["retrieval_k"]

        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")

        queries = state.get("search_queries", [state["input"]])
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

        tasks = []
        for i, q in enumerate(queries):
            tasks.append((f"bm25_{i}", _invoke_retriever(bm25, q, k_val)))
            tasks.append((f"faiss_{i}", _invoke_retriever(faiss_ret, q, k_val)))

        keys = [t[0] for t in tasks]
        coroutines = [t[1] for t in tasks]
        results_list = await asyncio.gather(*coroutines)

        search_results_map = {}
        for key, res in zip(keys, results_list, strict=False):
            if res:
                for doc in res:
                    if not doc.metadata.get("doc_id"):
                        doc.metadata["doc_id"] = fast_hash(doc.page_content)
                search_results_map[key] = res

        wrapped_results = {
            node_id: [
                AggregatedSearchResult(
                    doc_id=doc.metadata["doc_id"],
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
        aggregated, _ = aggregator.aggregate_results(
            wrapped_results,
            strategy=AggregationStrategy.RELATIVE_SCORE_FUSION,
            top_k=RERANKER_CONFIG.get("max_rerank_docs", 15),
        )

        final_docs = [
            Document(page_content=r.content, metadata=r.metadata) for r in aggregated
        ]
        SessionManager.replace_last_status_log(
            f"문서 {len(final_docs)}개 지능형 RRF 통합 완료"
        )
        return {"documents": final_docs}

    # 3. 재순위화 노드
    async def rerank_documents(state: GraphState) -> GraphOutput:
        documents = state.get("documents", [])
        if not RERANKER_CONFIG.get("enabled", True) or not documents:
            return {"documents": documents}
        if len(documents) <= 1:
            return {"documents": documents}

        route = state.get("route_decision", "FACTOID")
        params = INTENT_PARAMETERS.get(route, INTENT_PARAMETERS.get("DEFAULT"))

        try:
            reranker_model = load_reranker_model(RERANKER_CONFIG.get("model_name"))
            if not reranker_model:
                return {"documents": documents[: params["rerank_top_k"]]}

            target_docs = documents[:15]
            pairs = [[state["input"], doc.page_content] for doc in target_docs]

            if hasattr(reranker_model, "apredict"):
                scores = await reranker_model.apredict(pairs)
            else:
                scores = await asyncio.to_thread(reranker_model.predict, pairs)

            scored_docs = sorted(
                zip(target_docs, scores, strict=False), key=lambda x: x[1], reverse=True
            )

            final_docs = []
            min_score = params["rerank_threshold"]
            for doc, score in scored_docs:
                if score >= min_score or len(final_docs) < 2:
                    final_docs.append(doc)

            final_docs = final_docs[: params["rerank_top_k"]]
            SessionManager.replace_last_status_log(
                f"핵심 정보 {len(final_docs)}개 의도({route}) 맞춤 선별 완료"
            )
            return {"documents": final_docs}
        except Exception as e:
            logger.error(f"[RERANK] 실패: {e}")
            return {"documents": documents[:5]}

    # 4. 문서 채점 노드 (Self-Correction)
    async def grade_documents(state: GraphState, config: RunnableConfig) -> GraphOutput:
        configurable = config.get("configurable", {})
        llm = configurable.get("llm") or SessionManager.get("llm")
        documents = state.get("documents", [])

        if not llm or not documents:
            return {"relevant_docs": documents}

        # [고도화] 채점용 프롬프트 보강
        grade_instruction = (
            "당신은 검색 결과의 관련성을 평가하는 지능적인 분석관입니다.\n"
            "제공된 [문서]가 [사용자 질문]에 답변하는 데 직접적 또는 간접적으로 도움이 되는 정보라면 'yes'라고 하십시오.\n"
            "핵심 주제와 연관된 기술적 설명이 포함되어 있다면 가급적 'yes'를 선택하십시오.\n"
            "오직 한 단어('yes' 또는 'no')만 출력하십시오."
        )

        batch_inputs = [
            [
                {"role": "system", "content": grade_instruction},
                {
                    "role": "user",
                    "content": f"[질문]: {state['input']}\n\n[문서]: {doc.page_content[:1200]}",
                },
            ]
            for doc in documents
        ]

        try:
            results = await llm.abatch(batch_inputs, config=config)
            relevant_docs = [
                doc
                for doc, res in zip(documents, results, strict=False)
                if "yes" in res.content.lower()
            ]
            final = relevant_docs or documents[:1]
            SessionManager.replace_last_status_log(
                f"검증 완료: {len(final)}/{len(documents)}개 핵심 정보 선정"
            )
            return {"relevant_docs": final, "documents": final}
        except Exception:
            return {"relevant_docs": documents, "documents": documents}

    # 5. 컨텍스트 포맷팅 노드 (Reorder 전략 복원)
    async def format_context(state: GraphState) -> GraphOutput:
        documents = state.get("relevant_docs", [])
        if not documents:
            return {"context": ""}

        # Long Context Reorder 전략
        docs_copy = documents.copy()
        reordered_docs = []
        left = True
        while docs_copy:
            if left:
                reordered_docs.append(docs_copy.pop(0))
            else:
                reordered_docs.insert(0, docs_copy.pop(0))
            left = not left

        merged_docs = _merge_consecutive_chunks(reordered_docs)

        # [고도화] 토큰 예산 적용
        formatted = []
        current_tokens = 0
        safe_budget = 4000  # 보수적 예산

        for i, doc in enumerate(merged_docs):
            page = doc.metadata.get("page", "?")
            text = f"-- DOC {i + 1} (P{page}) --\n{doc.page_content}"
            toks = count_tokens_rough(text)
            if current_tokens + toks > safe_budget:
                break
            formatted.append(text)
            current_tokens += toks

        return {"context": "\n\n".join(formatted)}

    # 6. 답변 생성 노드 (의도 기반 프롬프트 복원)
    async def generate_response(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        configurable = config.get("configurable", {})
        sid = configurable.get("session_id")
        llm = configurable.get("llm") or SessionManager.get("llm", session_id=sid)

        try:
            route = state.get("route_decision", "FACTOID")

            # [복구] 의도에 따른 특화 프롬프트 선택
            intent_prompt = {
                "RESEARCH": RESEARCH_SYSTEM_PROMPT,
                "SUMMARY": RESEARCH_SYSTEM_PROMPT,
                "GREETING": GREETING_SYSTEM_PROMPT,
                "OUT_OF_CONTEXT": OUT_OF_CONTEXT_SYSTEM_PROMPT,
            }.get(route, FACTOID_SYSTEM_PROMPT)

            sys_prompt = f"{ANALYSIS_PROTOCOL}\n\n{intent_prompt}"
            human_content = QA_HUMAN_PROMPT.format(
                context=state.get("context", "근거를 찾을 수 없습니다."),
                input=state["input"],
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": human_content},
            ]

            SessionManager.add_status_log(f"답변 생성 중 ({route})...", session_id=sid)
            tracker = ResponsePerformanceTracker(state["input"], llm)

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
                "route_decision": route,
            }
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    # Workflow Definition
    workflow = StateGraph(GraphState)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("router", router)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        lambda s: END if s.get("is_cached") else "router",
        {END: END, "router": "router"},
    )
    workflow.add_conditional_edges(
        "router",
        lambda s: "generate_response"
        if s.get("route_decision") == "GREETING"
        else "generate_queries",
        {
            "generate_response": "generate_response",
            "generate_queries": "generate_queries",
        },
    )
    workflow.add_edge("generate_queries", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_documents")
    workflow.add_edge("grade_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
