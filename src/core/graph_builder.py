"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Core Logic Rebuild: 데코레이터 제거 및 순수 함수 구조로 변경하여 config 전달 보장.
"""

import logging
import asyncio
import hashlib
import re
import time
from contextlib import aclosing
from typing import Dict, List, Optional, overload, Any

from common.typing_utils import (
    DocumentList,
    GraphState,
    GraphOutput,
    T,
)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import StateGraph, END

from common.config import QA_SYSTEM_PROMPT, RERANKER_CONFIG, QUERY_EXPANSION_PROMPT, QUERY_EXPANSION_CONFIG
from common.constants import TimeoutConstants
from api.schemas import GraphState
from core.model_loader import load_reranker_model
from core.query_optimizer import RAGQueryOptimizer # 추가
from common.utils import clean_query_text
from services.monitoring.performance_monitor import get_performance_monitor, OperationType
from services.optimization.async_optimizer import (
    get_concurrent_query_expander,
    get_concurrent_document_retriever,
    get_concurrent_document_reranker,
    get_async_config
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

# --- Helper Functions ---

def _merge_consecutive_chunks(docs: DocumentList) -> DocumentList:
    """
    같은 출처의 연속된 인덱스를 가진 청크들을 하나로 병합합니다.
    [수정] 페이지 번호가 같은 경우에만 병합하여 출처 정확도를 높입니다.
    """
    if not docs:
        return []
        
    sorted_docs = sorted(
        docs, 
        key=lambda d: (
            d.metadata.get("source", ""), 
            d.metadata.get("page", 0),
            d.metadata.get("chunk_index", -1)
        )
    )
    
    merged = []
    if not sorted_docs:
        return merged

    current_doc = Document(
        page_content=sorted_docs[0].page_content,
        metadata=sorted_docs[0].metadata.copy()
    )
    
    for next_doc in sorted_docs[1:]:
        curr_src = current_doc.metadata.get("source")
        next_src = next_doc.metadata.get("source")
        curr_page = current_doc.metadata.get("page")
        next_page = next_doc.metadata.get("page")
        
        curr_idx = current_doc.metadata.get("chunk_index", -1)
        next_idx = next_doc.metadata.get("chunk_index", -1)
        
        # 같은 소스, 같은 페이지, 연속된 인덱스일 때만 병합
        if (curr_src == next_src and 
            curr_page == next_page and 
            curr_idx != -1 and next_idx == curr_idx + 1):
            
            current_doc.page_content += " " + next_doc.page_content
            # 인덱스 업데이트 (마지막 인덱스로)
            current_doc.metadata["chunk_index"] = next_idx
        else:
            merged.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content,
                metadata=next_doc.metadata.copy()
            )
    merged.append(current_doc)
    return merged


# --- Graph Construction ---

def build_graph(retriever: Optional[T] = None) -> T:
    """
    RAG 워크플로우를 구성하고 컴파일합니다.
    """
    # Backward-compat: allow tests/legacy code to call build_graph() without providing a retriever.
    # In that case, retrieval returns empty docs, and the graph still compiles.
    if retriever is None:
        class _EmptyRetriever:
            def invoke(self, _q: str):
                return []
            async def ainvoke(self, _q: str):
                return []
        retriever = _EmptyRetriever()

    # 1. 쿼리 확장 노드 (AsyncIO 최적화)
    async def generate_queries(state: GraphState, config: RunnableConfig) -> GraphOutput:
        """
        [AsyncIO 최적화] 쿼리 확장 시 동시 처리
        """
        from core.session import SessionManager
        with monitor.track_operation(OperationType.QUERY_PROCESSING, {"stage": "async_query_expansion"}) as op:
            logger.info(f"[Query] [Start] 질문 분석 및 검색어 확장 시작: '{state['input'][:50]}...'")
            
            # [최적화] 지능형 쿼리 라우팅
            if not RAGQueryOptimizer.is_complex_query(state["input"]):
                logger.info("[Query] [Routing] 단순 질문 감지 -> 확장 생략")
                SessionManager.replace_last_status_log("단순 질문 분석 완료")
                SessionManager.add_status_log("문서 검색 중")
                return {"search_queries": [state["input"]]}

            if not QUERY_EXPANSION_CONFIG.get("enabled", True):
                return {"search_queries": [state["input"]]}
        
        llm = config.get("configurable", {}).get("llm")
        if not llm:
            return {"search_queries": [state["input"]]}

        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_EXPANSION_PROMPT),
            ("human", "{input}"),
        ])
        chain = prompt | llm | StrOutputParser()
        
        try:
            # [최적화] 비동기 LLM 호출
            result = await chain.ainvoke({"input": state["input"]})
            
            # [개선] 불렛 포인트 및 번호 제거 파싱 로직
            raw_queries = [q.strip() for q in result.split("\n") if q.strip()]
            clean_queries = []
            for q in raw_queries:
                clean_q = clean_query_text(q)
                if clean_q:
                    clean_queries.append(clean_q)
            
            final_queries = clean_queries[:3] if clean_queries else [state["input"]]
            logger.info(f"[Query] [Complete] 검색어 {len(final_queries)}개 생성 완료: {final_queries}")
            SessionManager.replace_last_status_log(f"질문 정밀 분석 완료 (검색어 {len(final_queries)}개)")
            SessionManager.add_status_log("문서 검색 중")
            
            op.tokens = len(result.split())
            return {"search_queries": final_queries}

        except Exception as e:
            op.error = str(e)
            logger.warning(f"[Query] [Error] 쿼리 확장 실패 (폴백 적용): {e}")
            return {"search_queries": [state["input"]]}


    # 2. 문서 검색 노드 (AsyncIO 최적화)
    async def retrieve_documents(state: GraphState) -> GraphOutput:
        """
        [AsyncIO 최적화] 여러 쿼리로부터 병렬 문서 검색
        """
        from core.session import SessionManager
        with monitor.track_operation(
            OperationType.DOCUMENT_RETRIEVAL, 
            {"query_count": len(state.get("search_queries", [state["input"]]))}
        ) as op:
            queries = state.get("search_queries", [state["input"]])
            logger.info(f"[Retrieval] [Start] 병렬 검색 시작 ({len(queries)}개 검색어)")
        
            async def _safe_ainvoke_with_timeout(q: str) -> DocumentList:
                """리트리버를 타임아웃과 함께 호출합니다."""
                try:
                    timeout = TimeoutConstants.RETRIEVER_TIMEOUT
                    
                    if hasattr(retriever, "ainvoke"):
                        result = await asyncio.wait_for(
                            retriever.ainvoke(q),
                            timeout=timeout
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(retriever.invoke, q),
                            timeout=timeout
                        )
                    
                    return result if isinstance(result, list) else []
                
                except asyncio.TimeoutError:
                    logger.error(
                        f"[Retrieval] [Timeout] 검색 시간 초과 ({timeout}s): {q}"
                    )
                    return []
                except Exception as e:
                    logger.error(f"[Retrieval] [Error] 검색 오류 ({q}): {e}", exc_info=True)
                    return []

            # [최적화] ConcurrentDocumentRetriever 사용
            try:
                retriever_obj = get_concurrent_document_retriever()
                unique_docs, retrieval_stats = await retriever_obj.retrieve_documents_parallel(
                    queries=queries,
                    retriever_func=_safe_ainvoke_with_timeout,
                    deduplicate=True,
                    metadata={"pipeline": "concurrent"}
                )
                
                logger.info(
                    f"[Retrieval] [Complete] 검색 완료 | "
                    f"총 발견: {retrieval_stats['total_retrieved']} | "
                    f"중복 제거: {retrieval_stats['duplicates_removed']} | "
                    f"최종 문서: {retrieval_stats['unique_count']}"
                )
                SessionManager.replace_last_status_log(f"관련 문장 {retrieval_stats['unique_count']}개 식별 완료")
                SessionManager.add_status_log("핵심 문장 엄선 중 (Reranking)")
                op.tokens = sum(len(doc.page_content.split()) for doc in unique_docs)
                return {"documents": unique_docs}
            
            except Exception as e:
                logger.error(f"[Retrieval] [Error] 동시 검색 최적화 실패: {e}")
                # 폴백: 기본 병렬 검색
                results = await asyncio.gather(*[_safe_ainvoke_with_timeout(q) for q in queries])
                all_documents = [doc for sublist in results for doc in sublist]
                
                unique_docs = []
                seen = set()
                for doc in all_documents:
                    doc_key = doc.page_content + doc.metadata.get("source", "")
                    doc_hash = hashlib.sha256(doc_key.encode()).hexdigest()
                    if doc_hash not in seen:
                        unique_docs.append(doc)
                        seen.add(doc_hash)
                
                logger.info(f"[Retrieval] [Fallback] 폴백 검색 완료: {len(unique_docs)} 문서")
                op.tokens = sum(len(doc.page_content.split()) for doc in unique_docs)
                return {"documents": unique_docs}


    # 3. 재순위화 노드 (지능형 최적화)
    async def rerank_documents(state: GraphState) -> GraphOutput:
        """
        [지능형 최적화] 리랭킹 성능 및 자원 효율화
        """
        from core.session import SessionManager
        documents = state.get("documents", [])
        if not RERANKER_CONFIG.get("enabled", False) or not documents:
            logger.info("[Rerank] [Skip] 리랭킹 비활성화 또는 문서 없음")
            return {"documents": documents}

        with monitor.track_operation(
            OperationType.RERANKING, 
            {"doc_count": len(documents)}
        ) as op:
            try:
                # [최적화 1] 바이패스 로직
                if len(documents) <= 1:
                    logger.info("[Rerank] [Skip] 문서가 1개뿐이므로 생략")
                    return {"documents": documents}

                # [최적화 2] 대상 문서 제한 (VRAM 및 속도 보호)
                max_docs = RERANKER_CONFIG.get("max_rerank_docs", 12)
                if len(documents) > max_docs:
                    logger.info(f"[Rerank] [Limit] 대상 제한: {len(documents)} -> {max_docs}")
                    documents = documents[:max_docs]

                logger.info(f"[Rerank] [Start] 지능형 리랭킹 시작 ({len(documents)}개 문서)")
                reranker = load_reranker_model(RERANKER_CONFIG.get("model_name"))
                top_k = RERANKER_CONFIG.get("top_k", 6)
                
                async def _rerank_batch(query: str, docs: DocumentList) -> List[float]:
                    try:
                        pairs = [[query, doc.page_content] for doc in docs]
                        scores = await asyncio.to_thread(reranker.predict, pairs)
                        return scores
                    except Exception as e:
                        logger.error(f"[Rerank] [Error] 리랭킹 배치 오류: {e}")
                        return [0.0] * len(docs)
                
                # ConcurrentDocumentReranker 사용
                reranker_obj = get_concurrent_document_reranker()
                final_docs, rerank_stats = await reranker_obj.rerank_documents_parallel(
                    query=state["input"],
                    documents=documents,
                    reranker_func=_rerank_batch,
                    top_k=top_k,
                    metadata={"pipeline": "optimized_concurrent"}
                )
                
                logger.info(
                    f"[Rerank] [Complete] 리랭킹 완료: {rerank_stats['input_count']} -> {rerank_stats['output_count']} 문서 선정"
                )
                SessionManager.replace_last_status_log(f"핵심 문장 {rerank_stats['output_count']}개 엄선 완료")
                SessionManager.add_status_log("답변 작성 준비 중")

                op.tokens = sum(len(doc.page_content.split()) for doc in final_docs)
                return {"documents": final_docs}
            
            except Exception as e:
                op.error = str(e)
                logger.error(f"[Rerank] [Error] 리랭킹 실패 (폴백 적용): {e}")
                return {"documents": documents[:6]} # 실패 시 상위 6개 반환


    # 4. 컨텍스트 포맷팅 노드
    def format_context(state: GraphState) -> GraphOutput:
        documents = state["documents"]
        if not documents:
            return {"context": ""}

        merged_docs = _merge_consecutive_chunks(documents)
        formatted = []
        for i, doc in enumerate(merged_docs):
            page = doc.metadata.get("page", "?")
            # 모델이 출력 형식과 혼동하지 않도록 형식을 변경합니다.
            formatted.append(f"-- DOCUMENT CONTENT (PAGE {page}) --\n{doc.page_content}")
        
        context_str = "\n\n".join(formatted)
        return {"context": context_str}


    # 5. 답변 생성 노드 (가장 중요)
    async def generate_response(state: GraphState, config: RunnableConfig) -> GraphOutput:
        """
        LLM 답변을 생성하고 스트리밍합니다. 
        내부적으로 성능 지표(TTFT, 추론 시간 등)를 상세히 기록합니다.
        """
        from core.session import SessionManager
        from common.utils import get_ollama_resource_usage

        with monitor.track_operation(OperationType.LLM_INFERENCE, {"doc_count": len(state.get("documents", []))}) as op:
            try:
                llm = config.get("configurable", {}).get("llm")
                if not llm:
                    raise ValueError("LLM 인스턴스가 설정(config)에 누락되었습니다.")
                
                model_name = getattr(llm, "model", "Unknown")
                resource_status = get_ollama_resource_usage(model_name)
                
                logger.info(f"[LLM] [Start] 답변 생성 프로세스 시작 (Model: {model_name}, Resource: {resource_status})")
                SessionManager.add_status_log(f"답변 준비 중 ({resource_status})")
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", QA_SYSTEM_PROMPT),
                    ("human", """### [Context] ###
{context}

### [Question] ###
{input}

### [Execution Checklist] ###
1. 위 [Question]의 언어를 확인하고, 동일한 언어로 답변을 시작하세요.
2. 모든 정보 단위의 마지막에 [Context]에서 확인한 페이지 번호 인용(`[p.X]`)을 마침표 앞에 덧붙이세요.
3. 명확하고 구조화된 형식으로 답변을 구성하세요."""),
                ])

                generation_chain = prompt_template | llm

                class ResponsePerformanceTracker:
                    def __init__(self, query_text: str, model_instance: Any):
                        self.start_time = time.time()
                        self.query = query_text
                        self.model = model_instance
                        self.first_token_at = None
                        self.thinking_started_at = None
                        self.thinking_finished_at = None
                        self.answer_started_at = None
                        self.answer_finished_at = None
                        self.chunk_count = 0
                        self.full_response = ""
                        self.full_thought = ""
                        self._log_thinking_start = False
                        self._log_answer_start = False

                    def record_chunk(self, content: str, thought: str):
                        now = time.time()
                        if self.first_token_at is None:
                            self.first_token_at = now
                        
                        if thought:
                            if not self._log_thinking_start:
                                logger.info(f"[LLM] [Thinking] 사고 과정 시작...")
                                SessionManager.replace_last_status_log("사고 과정 기록 중...")
                                self.thinking_started_at = now
                                self._log_thinking_start = True
                            self.full_thought += thought
                            
                        if content:
                            if not self._log_answer_start:
                                # 답변이 시작되면 사고 과정은 종료된 것으로 간주
                                if self.thinking_started_at and self.thinking_finished_at is None:
                                    self.thinking_finished_at = now
                                    thinking_dur = self.thinking_finished_at - self.thinking_started_at
                                    logger.info(f"[LLM] [Thinking] 사고 완료 ({thinking_dur:.2f}s)")
                                
                                logger.info(f"[LLM] [Response] 답변 스트리밍 시작")
                                SessionManager.replace_last_status_log("답변 스트리밍 중...")
                                self.answer_started_at = now
                                self._log_answer_start = True
                            
                            self.full_response += content
                            self.chunk_count += 1

                    def finalize_and_log(self):
                        self.answer_finished_at = time.time()
                        
                        total_duration = self.answer_finished_at - self.start_time
                        time_to_first_token = (self.first_token_at - self.start_time) if self.first_token_at else 0
                        
                        # 사고 과정 시간 계산
                        thinking_duration = 0
                        if self.thinking_started_at:
                            end_time = self.thinking_finished_at or self.answer_started_at or self.answer_finished_at
                            thinking_duration = end_time - self.thinking_started_at
                            
                        # 실제 답변 생성 시간 계산
                        answer_duration = (self.answer_finished_at - self.answer_started_at) if self.answer_started_at else 0
                        
                        # 토큰 수 및 속도 계산
                        resp_token_count = len(self.full_response.split())
                        thought_token_count = len(self.full_thought.split())
                        tokens_per_second = resp_token_count / answer_duration if answer_duration > 0 else 0
                        
                        # 표준화된 상세 로그 기록
                        logger.info(
                            f"[LLM] [Complete] 생성 완료 | "
                            f"TTFT: {time_to_first_token:.2f}s | "
                            f"사고: {thinking_duration:.2f}s ({thought_token_count} tok) | "
                            f"답변: {answer_duration:.2f}s ({resp_token_count} tok) | "
                            f"속도: {tokens_per_second:.1f} tok/s | "
                            f"총 소요: {total_duration:.2f}s"
                        )

                        # UI 상태 박스 최종 업데이트
                        SessionManager.replace_last_status_log(
                            f"답변 생성 완료 (사고: {thought_token_count}토큰, 답변: {resp_token_count}토큰)"
                        )

                        # CSV 기록
                        try:
                            monitor.log_to_csv({
                                "model": getattr(self.model, "model", "unknown"),
                                "ttft": time_to_first_token,
                                "thinking": thinking_duration,
                                "answer": answer_duration,
                                "total": total_duration,
                                "tokens": resp_token_count,
                                "thought_tokens": thought_token_count,
                                "tps": tokens_per_second,
                            })
                        except: pass

                        return resp_token_count, thought_token_count

                tracker = ResponsePerformanceTracker(state["input"], llm)
                timeout = TimeoutConstants.LLM_TIMEOUT

                async def _consume_stream_and_dispatch_events() -> None:
                    """모델 스트림을 소비하고 가공하여 커스텀 이벤트를 발생시킵니다."""
                    answer_buffer = []
                    buffer_size = 5  # 5토큰 단위로 묶어서 전송
                    
                    async for chunk in generation_chain.astream(
                        {"input": state["input"], "context": state["context"]},
                        config=config,
                    ):
                        content = getattr(chunk, "content", "")
                        thought = ""
                        if hasattr(chunk, "additional_kwargs"):
                            thought = chunk.additional_kwargs.get("thought") or chunk.additional_kwargs.get("thinking", "")
                        
                        if not content and not thought:
                            continue

                        tracker.record_chunk(content, thought)

                        # 사고 과정(thought)은 내용이 있을 때 즉시 전송 (보통 연속해서 들어옴)
                        if thought:
                            await adispatch_custom_event(
                                "response_chunk",
                                {"chunk": "", "thought": thought},
                                config=config,
                            )
                        
                        # 답변 본문(content)은 버퍼링하여 전송 횟수 최적화
                        if content:
                            answer_buffer.append(content)
                            if len(answer_buffer) >= buffer_size:
                                await adispatch_custom_event(
                                    "response_chunk",
                                    {"chunk": "".join(answer_buffer), "thought": ""},
                                    config=config,
                                )
                                answer_buffer = []

                    # 스트리밍 종료 후 남은 버퍼 플러시
                    if answer_buffer:
                        await adispatch_custom_event(
                            "response_chunk",
                            {"chunk": "".join(answer_buffer), "thought": ""},
                            config=config,
                        )

                # 스트리밍 태스크 실행
                generation_task = asyncio.create_task(_consume_stream_and_dispatch_events())

                try:
                    await asyncio.wait_for(generation_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"[LLM] [Error] 응답 생성 시간 초과 ({timeout}초)")
                    generation_task.cancel()
                    try: await generation_task
                    except asyncio.CancelledError: pass
                    if not tracker.full_response:
                        tracker.full_response = "죄송합니다. 응답 생성 시간이 너무 오래 걸려 중단되었습니다."
                    op.error = "timeout"
                except Exception as e:
                    if not generation_task.done():
                        generation_task.cancel()
                    logger.error(f"[LLM] [Error] 응답 생성 중 예외 발생: {e}", exc_info=True)
                    op.error = str(e)
                    raise
                
                # 최종 지표 기록
                resp_tokens, thought_tokens = tracker.finalize_and_log()
                op.tokens = resp_tokens
                op.metadata["thought_tokens"] = thought_tokens
                
                return {
                    "response": tracker.full_response, 
                    "thought": tracker.full_thought,
                    "documents": state.get("documents", [])
                }
            except Exception as e:
                logger.error(f"[Graph] [Critical] generate_response 노드 치명적 오류: {e}")
                raise


    # --- Workflow Definition ---
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("generate_queries")
    workflow.add_edge("generate_queries", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
