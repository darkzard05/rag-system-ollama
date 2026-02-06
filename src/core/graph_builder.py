"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Core Logic Rebuild: 데코레이터 제거 및 순수 함수 구조로 변경하여 config 전달 보장.
"""

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from api.schemas import GraphState
from common.config import (
    ANALYSIS_PROTOCOL,
    QUERY_EXPANSION_CONFIG,
    QUERY_EXPANSION_PROMPT,
    RERANKER_CONFIG,
)
from common.constants import TimeoutConstants
from common.typing_utils import (
    DocumentList,
    GraphOutput,
    T,
)
from core.model_loader import load_reranker_model
from services.monitoring.performance_monitor import (
    OperationType,
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


def _get_intent_params(intent: str | None) -> dict[str, Any]:
    """의도별 동적 파이프라인 파라미터를 결정합니다."""
    params = {
        "FACTOID": {
            "retrieval_k": 10,
            "rerank_top_k": 5,
            "rerank_threshold": 0.5,
            "sort_by_page": False,
            "use_strict_grading": True,
        },
        "SUMMARY": {
            "retrieval_k": 40,
            "rerank_top_k": 20,
            "rerank_threshold": 0.15,
            "sort_by_page": True,
            "use_strict_grading": False,
        },
        "RESEARCH": {
            "retrieval_k": 25,
            "rerank_top_k": 10,
            "rerank_threshold": 0.3,
            "sort_by_page": False,
            "use_strict_grading": False,
        },
    }
    return params.get(
        intent or "",
        {
            "retrieval_k": 20,
            "rerank_top_k": 7,
            "rerank_threshold": 0.35,
            "sort_by_page": False,
            "use_strict_grading": False,
        },
    )


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


# --- Helper Classes ---


class ResponsePerformanceTracker:
    def __init__(self, query_text: str, model_instance: Any):
        from core.session import SessionManager

        self.SessionManager = SessionManager
        self.start_time: float = time.time()
        self.query: str = query_text
        self.model: Any = model_instance
        self.first_token_at: float | None = None
        self.thinking_started_at: float | None = None
        self.thinking_finished_at: float | None = None
        self.answer_started_at: float | None = None
        self.answer_finished_at: float | None = None
        self.chunk_count: int = 0
        self._resp_parts: list[str] = []
        self._thought_parts: list[str] = []
        self.full_response: str = ""
        self.full_thought: str = ""
        self._log_thinking_start: bool = False
        self._log_answer_start: bool = False

    def record_chunk(self, content: str, thought: str):
        now = time.time()
        if self.first_token_at is None:
            self.first_token_at = now

        if thought:
            if not self._log_thinking_start:
                self.SessionManager.replace_last_status_log("사고 과정 기록 중...")
                self.thinking_started_at = now
                self._log_thinking_start = True
            self._thought_parts.append(thought)

        if content:
            # [최적화] 중복 기록 방지 (이미 버퍼에 있다면 무시)
            if (
                not self._resp_parts or self._resp_parts[-1] != content
            ) and not self._log_answer_start:
                if self.thinking_started_at and self.thinking_finished_at is None:
                    self.thinking_finished_at = now

                self.SessionManager.replace_last_status_log("답변 스트리밍 중...")
                self.answer_started_at = now
                self._log_answer_start = True

            if not self._resp_parts or self._resp_parts[-1] != content:
                self._resp_parts.append(content)
                self.chunk_count += 1

    def finalize_and_log(self) -> Any:
        from api.schemas import PerformanceStats

        self.answer_finished_at = time.time()
        self.full_response = "".join(self._resp_parts)
        self.full_thought = "".join(self._thought_parts)

        total_duration = self.answer_finished_at - self.start_time
        time_to_first_token = (
            (self.first_token_at - self.start_time) if self.first_token_at else 0
        )

        thinking_duration: float = 0.0
        if self.thinking_started_at:
            end_time = (
                self.thinking_finished_at
                or self.answer_started_at
                or self.answer_finished_at
            )
            thinking_duration = end_time - self.thinking_started_at

        answer_duration: float = (
            (self.answer_finished_at - self.answer_started_at)
            if self.answer_started_at
            else 0.0
        )
        # [수정] split() 대신 유틸리티 함수를 사용하여 더 정확한 토큰 수 추정
        from common.utils import count_tokens_rough

        resp_token_count = count_tokens_rough(self.full_response)
        thought_token_count = count_tokens_rough(self.full_thought)

        tokens_per_second: float = (
            (resp_token_count / answer_duration) if answer_duration > 0 else 0.0
        )

        stats = PerformanceStats(
            ttft=time_to_first_token,
            thinking_time=thinking_duration,
            generation_time=answer_duration,
            total_time=total_duration,
            token_count=resp_token_count,
            thought_token_count=thought_token_count,
            tps=tokens_per_second,
            model_name=getattr(self.model, "model", "unknown"),
        )

        logger.info(
            f"[LLM] 완료 | TTFT: {stats.ttft:.2f}s | "
            f"사고: {stats.thinking_time:.2f}s | 답변: {stats.generation_time:.2f}s | "
            f"속도: {stats.tps:.1f} tok/s"
        )

        self.SessionManager.replace_last_status_log(
            f"완료 (사고 {stats.thought_token_count} / 답변 {stats.token_count})"
        )

        with contextlib.suppress(Exception):
            monitor.log_to_csv(
                {
                    "model": stats.model_name,
                    "ttft": stats.ttft,
                    "thinking": stats.thinking_time,
                    "answer": stats.generation_time,
                    "total": stats.total_time,
                    "tokens": stats.token_count,
                    "thought_tokens": stats.thought_token_count,
                    "tps": stats.tps,
                }
            )
        return stats


# --- Graph Construction ---


def build_graph(retriever: T | None = None) -> T:
    """RAG 워크플로우를 구성하고 컴파일합니다."""
    if retriever is None:

        class _EmptyRetriever:
            def invoke(self, _q: str):
                return []

            async def ainvoke(self, _q: str):
                return []

        retriever = _EmptyRetriever()

    # 0. 의도 분석 노드 (NEW)
    async def router(state: GraphState, config: RunnableConfig) -> GraphOutput:
        from core.query_optimizer import RAGQueryOptimizer
        from core.session import SessionManager

        llm = config.get("configurable", {}).get("llm") or SessionManager.get("llm")
        if not llm:
            return {"route_decision": "FACTOID"}

        intent = await RAGQueryOptimizer.classify_intent(state["input"], llm)
        SessionManager.replace_last_status_log(f"의도 분석 완료: {intent}")
        return {"route_decision": intent}

    # 1. 쿼리 확장 노드 (의도 기반 지능형 쿼리 생성)
    async def generate_queries(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        from core.session import SessionManager

        route = state.get("route_decision", "FACTOID")

        # 인사말이면 쿼리 확장 불필요
        if route == "GREETING":
            return {"search_queries": []}

        with monitor.track_operation(
            OperationType.QUERY_PROCESSING, {"stage": "structured_query_expansion"}
        ):
            if not QUERY_EXPANSION_CONFIG.get("enabled", True):
                return {"search_queries": [state["input"]]}

        llm = config.get("configurable", {}).get("llm") or SessionManager.get("llm")
        if not llm:
            return {"search_queries": [state["input"]]}

        # [최적화] 의도에 따른 검색 전략 차별화
        if route == "SUMMARY":
            # [수정] 사용자 질문의 핵심 주제를 반영하도록 지시문 강화
            instruction = (
                f"사용자가 다음 문서에 대한 전체 요약 또는 재구성을 요청했습니다: '{state['input']}'\n"
                "문서의 핵심 골격을 파악하기 위해 다음 3가지 관점의 고유명사를 포함한 검색어를 생성하십시오:\n"
                "1. 해당 주제의 서론(Introduction) 및 초록(Abstract)\n"
                "2. 해당 주제의 주요 결론(Conclusion) 및 핵심 결과\n"
                "3. 이 문서만의 독창적인 기여도(Contributions) 및 핵심 아키텍처\n"
                "반드시 아래 JSON 형식을 엄격히 지켜 답변하십시오:\n"
                '{{"queries": ["keyword1", "keyword2", "keyword3"]}}'
            )
        else:
            doc_lang = SessionManager.get("doc_language") or "English"
            instruction = (
                f"{QUERY_EXPANSION_PROMPT}\n"
                "반드시 아래 JSON 형식을 엄격히 지켜 답변하십시오:\n"
                '{{"queries": ["키워드1", "키워드2", "키워드3"]}}'
            )
            if doc_lang == "English":
                instruction += "\n주의: 대상 문서가 영어이므로 검색 효율을 위해 확장 쿼리 중 2개 이상은 반드시 영어로 작성하십시오."

        try:
            bound_llm = llm.bind(format="json") if hasattr(llm, "bind") else llm
            prompt = ChatPromptTemplate.from_messages(
                [("system", instruction), ("human", "{input}")]
            )
            chain = prompt | bound_llm | StrOutputParser()

            response = await chain.ainvoke({"input": state["input"]})

            # 파싱 및 정제
            clean_response = response.strip()
            if "```" in clean_response:
                clean_response = (
                    clean_response.split("```")[1].replace("json", "").strip()
                )

            data = json.loads(clean_response)
            raw_queries = data.get("queries", [])

            clean_queries = [state["input"]]
            for q in raw_queries:
                q = q.strip().replace('"', "").replace("'", "")
                if q and len(q) > 2 and q not in clean_queries:
                    clean_queries.append(q)

            final_queries = clean_queries[:4]
            logger.info(f"[Query] 의도({route}) 기반 확장 완료: {final_queries}")
            SessionManager.replace_last_status_log(
                f"의도({route}) 기반 검색 전략 수립 완료"
            )
            return {"search_queries": final_queries}

        except Exception as e:
            logger.error(f"[Query] 확장 오류: {e}")
            return {"search_queries": [state["input"]]}

    # 2. 문서 검색 노드 (의도 기반 동적 K값 적용)
    async def retrieve_documents(state: GraphState) -> GraphOutput:
        from core.search_aggregator import AggregationStrategy, SearchResultAggregator
        from core.session import SessionManager

        route = state.get("route_decision", "FACTOID")
        params = _get_intent_params(route)
        k_val = params["retrieval_k"]

        with monitor.track_operation(
            OperationType.DOCUMENT_RETRIEVAL,
            {
                "query_count": len(state.get("search_queries", [state["input"]])),
                "intent": route,
            },
        ):
            queries = state.get("search_queries", [state["input"]])
            logger.info(
                f"[RETRIEVAL] 병렬 검색 시작 | 의도: {route} | K: {k_val} | 쿼리수: {len(queries)}"
            )
            bm25 = SessionManager.get("bm25_retriever")
            faiss_ret = SessionManager.get("faiss_retriever")

            async def _invoke_retriever(ret, q: str, k: int) -> DocumentList:
                if not ret:
                    return []
                try:
                    # 동적으로 k값 조정
                    if hasattr(ret, "k"):
                        ret.k = k
                    if hasattr(ret, "search_kwargs"):
                        ret.search_kwargs["k"] = k
                        ret.search_kwargs["filter"] = {"is_content": True}

                    if hasattr(ret, "ainvoke"):
                        return await asyncio.wait_for(
                            ret.ainvoke(q), timeout=TimeoutConstants.RETRIEVER_TIMEOUT
                        )
                    return await asyncio.to_thread(ret.invoke, q)
                except Exception as e:
                    logger.warning(f"리트리버 호출 실패: {e}")
                    return []

            # 모든 쿼리에 대해 리트리버 호출 태스크 생성
            search_results_map = {}
            tasks = []

            for i, q in enumerate(queries):
                tasks.append((f"bm25_{i}", _invoke_retriever(bm25, q, k_val)))
                tasks.append((f"faiss_{i}", _invoke_retriever(faiss_ret, q, k_val)))

            # 병렬 실행
            keys = [t[0] for t in tasks]
            coroutines = [t[1] for t in tasks]
            results_list = await asyncio.gather(*coroutines)

            # Aggregator 형식에 맞게 변환
            for key, res in zip(keys, results_list, strict=False):
                if res:
                    for doc in res:
                        # LangChain Document는 metadata에 점수를 저장하는 것이 안전함
                        if "score" not in doc.metadata:
                            doc.metadata["score"] = 0.5

                        if not doc.metadata.get("doc_id"):
                            from common.utils import fast_hash

                            key_str = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}_{doc.metadata.get('chunk_index')}"
                            doc.metadata["doc_id"] = fast_hash(key_str)

                        # Aggregator가 기대하는 속성을 동적으로 연결 (Pydantic 필드 제약 우회)
                        # r.score, r.doc_id, r.node_id 등에 대응하기 위해 dict 형태의 래퍼 사용 고려 가능하나
                        # 여기서는 SearchResultAggregator가 metadata를 직접 보게 하거나 래퍼 클래스 생성
                        pass
                    search_results_map[key] = res

            # [최적화] SearchResultAggregator가 Document 객체의 metadata를 인식하도록 래퍼 사용
            class ResultWrapper:
                def __init__(self, doc, node_id):
                    self.doc_id = doc.metadata["doc_id"]
                    self.content = doc.page_content
                    self.score = doc.metadata["score"]
                    self.node_id = node_id
                    self.metadata = doc.metadata

            wrapped_results = {
                node_id: [ResultWrapper(doc, node_id.split("_")[0]) for doc in docs]
                for node_id, docs in search_results_map.items()
            }

            aggregator = SearchResultAggregator()
            max_docs = RERANKER_CONFIG.get("max_rerank_docs", 15)

            aggregated, _metrics = aggregator.aggregate_results(
                wrapped_results, strategy=AggregationStrategy.RRF_FUSION, top_k=max_docs
            )

            # AggregatedResult를 다시 Document 객체로 변환
            final_docs = [
                Document(
                    page_content=r.content,
                    metadata={**r.metadata, "score": r.aggregated_score},
                )
                for r in aggregated
            ]

            SessionManager.replace_last_status_log(
                f"문서 {len(final_docs)}개 지능형 RRF 통합 완료"
            )
            SessionManager.add_status_log("핵심 문장 선별 중...")
            return {"documents": final_docs}

    async def _rerank_worker(
        query: str,
        docs: DocumentList,
        reranker: Any,
    ) -> list[float]:
        try:
            pairs = [[query, doc.page_content] for doc in docs]
            # [최적화] 네이티브 비동기 메서드가 있으면 직접 호출, 없으면 스레드 풀 활용
            if hasattr(reranker, "apredict"):
                return await reranker.apredict(pairs)
            return await asyncio.to_thread(reranker.predict, pairs)
        except Exception as e:
            logger.error(f"[Rerank] 오류: {e}")
            return [0.0] * len(docs)

    # 3. 재순위화 노드 (의도 기반 동적 정제)
    async def rerank_documents(state: GraphState) -> GraphOutput:
        from core.session import SessionManager

        documents = state.get("documents", [])
        if not RERANKER_CONFIG.get("enabled", True) or not documents:
            return {"documents": documents}
        if len(documents) <= 1:
            return {"documents": documents}

        route = state.get("route_decision", "FACTOID")
        params = _get_intent_params(route)

        with monitor.track_operation(
            OperationType.RERANKING, {"doc_count": len(documents), "intent": route}
        ):
            try:
                # 1. 1차 후보군 설정 (의도에 따라 유연하게)
                max_docs = params["retrieval_k"]  # 검색량에 맞춰 재순위 대상 확대
                target_docs = documents[:max_docs]

                reranker_model = load_reranker_model(RERANKER_CONFIG.get("model_name"))
                if not reranker_model:
                    return {"documents": target_docs[: params["rerank_top_k"]]}

                scores = await _rerank_worker(
                    state["input"], target_docs, reranker_model
                )
                scored_docs = []
                min_score = params["rerank_threshold"]

                logger.info(
                    f"=== [Intent-Aware Reranking: {route}] (Min: {min_score}) ==="
                )
                for doc, score in zip(target_docs, scores, strict=False):
                    scored_docs.append((doc, score))

                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # 2. 합격 필터링
                final_docs: list[Document] = []
                if scored_docs:
                    top_score = scored_docs[0][1]
                    # 동적 하한선: 최고 점수 기반 또는 고정 임계값 중 유연한 것 선택
                    effective_min = (
                        min(min_score, top_score * 0.6) if top_score > 0 else 0
                    )

                    for doc, score in scored_docs:
                        if (
                            score >= effective_min or len(final_docs) < 3
                        ):  # 최소 3개는 보장
                            final_docs.append(doc)

                # 3. 최종 Top-K 적용
                top_k = params["rerank_top_k"]
                final_docs = final_docs[:top_k]

                logger.info(
                    f"[RERANK] 완료: {len(target_docs)} -> {len(final_docs)} (의도: {route})"
                )
                SessionManager.replace_last_status_log(
                    f"핵심 정보 {len(final_docs)}개 의도({route}) 맞춤 선별 완료"
                )
                return {"documents": final_docs}
            except Exception as e:
                logger.error(f"[RERANK] 실패: {e}")
                return {"documents": documents[:5]}

    # 3. 문서 채점 노드 (Self-Correction)
    async def grade_documents(state: GraphState, config: RunnableConfig) -> GraphOutput:
        from core.session import SessionManager

        llm = config.get("configurable", {}).get("llm")
        documents = state.get("documents", [])

        if not llm or not documents:
            return {"relevant_docs": documents}

        logger.info(
            f"=== [Self-Correction] 문서 {len(documents)}개 관련성 검증 시작 ==="
        )
        SessionManager.add_status_log(f"검색된 문서 {len(documents)}개 정밀 검증 중...")

        # 채점용 프롬프트 (정확도 향상을 위해 지능적 판단 유도)
        grade_instruction = (
            "당신은 검색 결과의 관련성을 평가하는 지능적인 분석관입니다.\n"
            "제공된 [문서]가 [사용자 질문]에 답변하는 데 직접적 또는 간접적으로 도움이 되는 정보라면 'yes'라고 하십시오.\n"
            "특히 질문의 핵심 주제(예: 학습 방법, 이미지 처리, 구조 등)와 연관된 기술적 설명이 포함되어 있다면 가급적 'yes'를 선택하십시오.\n"
            "단, 본문이 전혀 없는 단순 참고문헌 리스트나 목차 페이지인 경우에만 'no'라고 하십시오.\n"
            "다른 설명 없이 오직 한 단어('yes' 또는 'no')만 출력하십시오."
        )

        # [최적화] 배치 처리를 위한 입력 메시지 리스트 구성
        batch_inputs = []
        for doc in documents:
            prompt = f"[사용자 질문]: {state['input']}\n\n[문서 내용]: {doc.page_content[:1200]}"
            batch_inputs.append(
                [
                    {"role": "system", "content": grade_instruction},
                    {"role": "user", "content": prompt},
                ]
            )

        try:
            # [최적화 핵심] 개별 호출 대신 abatch() 활용하여 오버헤드 최소화
            # config 전달을 통해 동일한 세션/설정 유지
            results = await llm.abatch(batch_inputs, config=config)

            relevant_docs = []
            for doc, res in zip(documents, results, strict=False):
                score = res.content.strip().lower()
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content[:60].replace("\n", " ")

                if "yes" in score:
                    relevant_docs.append(doc)
                    logger.info(f"└─ [PASS] P{page} | {snippet}...")
                else:
                    logger.info(f"└─ [FAIL] P{page} | {snippet}...")

        except Exception as e:
            logger.error(f"문서 배치 채점 오류: {e}")
            relevant_docs = documents[:2]  # 오류 시 보수적 복구

        # 만약 모든 문서가 탈락하면, 아무것도 안 주는 대신 '가장 가능성 있는' 1개만 살림 (노이즈 최소화)
        if not relevant_docs and documents:
            logger.warning(
                "[Self-Correction] 모든 문서 부적합. 가장 유사도가 높은 1개만 제한적 복구."
            )
            relevant_docs = documents[:1]

        SessionManager.replace_last_status_log(
            f"검증 완료: {len(relevant_docs)}/{len(documents)}개 핵심 정보 선정"
        )
        # [최적화] 메인 문서 리스트를 교정된 리스트로 업데이트하여 이후 단계 및 UI와의 일관성 보장
        return {"relevant_docs": relevant_docs, "documents": relevant_docs}

    # 4. 컨텍스트 포맷팅 노드 (의도 기반 정렬 최적화)
    async def format_context(state: GraphState, config: RunnableConfig) -> GraphOutput:
        from common.utils import count_tokens_rough

        documents = state.get("relevant_docs", [])
        if not documents:
            logger.warning("[CHAT] [CONTEXT] 컨텍스트가 비어있습니다.")
            return {"context": ""}

        route = state.get("route_decision", "FACTOID")
        params = _get_intent_params(route)

        # --- [의도 기반 정렬 전략] ---
        reordered_docs = []
        if params["sort_by_page"]:
            # [SUMMARY 전용] 페이지 번호와 청크 인덱스 순으로 정렬하여 서술 흐름 복원
            logger.info(
                f"=== [Context Strategy: Page-Sort] 의도({route}) 흐름 유지 ==="
            )
            reordered_docs = sorted(
                documents,
                key=lambda d: (
                    d.metadata.get("page", 0),
                    d.metadata.get("chunk_index", 0),
                ),
            )
        else:
            # [RESEARCH/FACTOID] Long Context Reorder (양 끝단 중요도 강화)
            logger.info(
                f"=== [Context Strategy: Reorder] 의도({route}) 정보 배치 최적화 ==="
            )
            docs_copy = documents.copy()
            left = True
            while docs_copy:
                if left:
                    reordered_docs.append(docs_copy.pop(0))
                else:
                    reordered_docs.insert(0, docs_copy.pop(0))
                left = not left
        # ---------------------------

        merged_docs = _merge_consecutive_chunks(reordered_docs)

        llm = config.get("configurable", {}).get("llm")
        num_ctx = getattr(llm, "num_ctx", 4096)
        safe_budget = int(num_ctx * 0.6)  # 예산 소폭 상향
        formatted = []
        current_tokens = 0

        logger.info(f"=== [LLM Context Inject] (Budget: {safe_budget}) ===")
        for i, doc in enumerate(merged_docs):
            page = doc.metadata.get("page", "?")
            src = doc.metadata.get("source", "unknown")
            text = (
                f"-- DOCUMENT {i + 1} (SRC: {src}, PAGE: {page}) --\n{doc.page_content}"
            )
            toks = count_tokens_rough(text)
            if current_tokens + toks > safe_budget:
                logger.info(f"└─ [SKIP] Budget Exceeded at Doc {i + 1}")
                break
            formatted.append(text)
            current_tokens += toks
            snippet = doc.page_content.replace("\n", " ")[:80]
            logger.info(
                f"└─ [Inject] Chunk {i + 1} | P{page} | {toks}tok | {snippet}..."
            )

        return {"context": "\n\n".join(formatted)}

    # 5. 답변 생성 노드 (생략...)

    # 5. 답변 생성 노드 (Prompt Caching 지원)
    async def generate_response(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        from common.utils import get_ollama_resource_usage
        from core.model_loader import ModelManager
        from core.session import SessionManager

        docs = state.get("documents") or []
        semaphore = ModelManager.get_inference_semaphore()

        async with semaphore:
            with monitor.track_operation(
                OperationType.LLM_INFERENCE, {"doc_count": len(docs)}
            ):
                try:
                    llm = config.get("configurable", {}).get("llm")
                    if not llm:
                        raise ValueError("LLM missing in config")

                    model_name = getattr(llm, "model", "Unknown")
                    logger.info(
                        f"[CHAT] [LLM] 생성 시작 | 모델: {model_name} | 자원: {get_ollama_resource_usage(model_name)}"
                    )
                    SessionManager.add_status_log("답변 생성 시작")

                    # [최적화] 시스템 프롬프트 구성 (중복 레이블 제거 및 구조화)
                    from common.config import (
                        FACTOID_SYSTEM_PROMPT,
                        GREETING_SYSTEM_PROMPT,
                        OUT_OF_CONTEXT_SYSTEM_PROMPT,
                        QA_HUMAN_PROMPT,
                        RESEARCH_SYSTEM_PROMPT,
                    )

                    # 1. 의도에 따른 특화 프롬프트 선택
                    route_decision = state.get("route_decision", "FACTOID")
                    intent_prompt = ""
                    if route_decision in ["RESEARCH", "SUMMARY"]:
                        intent_prompt = RESEARCH_SYSTEM_PROMPT
                    elif route_decision == "GREETING":
                        intent_prompt = GREETING_SYSTEM_PROMPT
                    elif route_decision == "OUT_OF_CONTEXT":
                        intent_prompt = OUT_OF_CONTEXT_SYSTEM_PROMPT
                    else:
                        intent_prompt = FACTOID_SYSTEM_PROMPT

                    # 2. 통합 시스템 메시지 생성 (서두의 중복성 제거)
                    sys_prompt = f"{ANALYSIS_PROTOCOL}\n\n{intent_prompt}"

                    # 3. 휴먼 메시지 생성 (config.yml의 구조화된 템플릿 사용)
                    human_content = QA_HUMAN_PROMPT.format(
                        context=state.get("context", "관련 근거를 찾을 수 없습니다."),
                        input=state["input"],
                    )

                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": human_content},
                    ]
                    options = {
                        "temperature": getattr(llm, "temperature", 0.1),
                        "num_predict": getattr(llm, "num_predict", -1),
                        "top_p": getattr(llm, "top_p", 0.8),
                        "num_ctx": getattr(llm, "num_ctx", 4096),
                    }

                    tracker = ResponsePerformanceTracker(state["input"], llm)

                    async def _consume_stream():
                        client = ModelManager.get_async_client(
                            host=getattr(llm, "base_url", "http://127.0.0.1:11434")
                        )
                        buffer = []
                        total_chunks = 0
                        async for part in await client.chat(
                            model=model_name,
                            messages=messages,
                            stream=True,
                            options=options,
                        ):
                            msg = part.get("message", {})
                            content, thought = (
                                msg.get("content", ""),
                                msg.get("thinking", ""),
                            )
                            if thought:
                                tracker.record_chunk("", thought)
                                await adispatch_custom_event(
                                    "response_chunk",
                                    {"chunk": "", "thought": thought},
                                    config=config,
                                )
                            if content:
                                total_chunks += 1
                                if total_chunks <= 5:
                                    tracker.record_chunk(content, "")
                                    await adispatch_custom_event(
                                        "response_chunk",
                                        {"chunk": content, "thought": ""},
                                        config=config,
                                    )
                                else:
                                    buffer.append(content)
                                    if len(buffer) >= 3:
                                        merged = "".join(buffer)
                                        tracker.record_chunk(merged, "")
                                        await adispatch_custom_event(
                                            "response_chunk",
                                            {"chunk": merged, "thought": ""},
                                            config=config,
                                        )
                                        buffer = []
                        if buffer:
                            merged = "".join(buffer)
                            tracker.record_chunk(merged, "")
                            await adispatch_custom_event(
                                "response_chunk",
                                {"chunk": merged, "thought": ""},
                                config=config,
                            )

                    gen_task = asyncio.create_task(_consume_stream())
                    await asyncio.wait_for(
                        gen_task, timeout=TimeoutConstants.LLM_TIMEOUT
                    )

                    # [최적화] 반환 전에 통계 및 전체 답변 확정 (순서 중요)
                    stats = tracker.finalize_and_log()

                    return {
                        "response": tracker.full_response,
                        "thought": tracker.full_thought,
                        "documents": docs,
                        "performance": stats.model_dump(),
                        "route_decision": route_decision,
                    }

                except Exception as e:
                    logger.error(f"[Graph] generate_response 오류: {e}", exc_info=True)
                    raise

    # --- Workflow Definition ---
    workflow = StateGraph(GraphState)

    # 1. 노드 등록
    workflow.add_node("router", router)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    # 2. 엣지 연결 (조건부 라우팅 적용)
    workflow.add_edge(START, "router")

    # 라우팅 로직: GREETING이면 검색 생략
    def route_after_analysis(state: GraphState):
        if state.get("route_decision") == "GREETING":
            return "generate_response"
        return "generate_queries"

    workflow.add_conditional_edges(
        "router",
        route_after_analysis,
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
