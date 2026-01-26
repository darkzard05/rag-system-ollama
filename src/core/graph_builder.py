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
        - 단일 쿼리를 다양한 관점으로 확장
        - 비동기 LLM 호출로 병렬 처리
        """
        with monitor.track_operation(OperationType.QUERY_PROCESSING, {"stage": "async_query_expansion"}) as op:
            logger.info("[Graph] 쿼리 생성 시작 (AsyncIO 최적화)")
            
            # [최적화] 지능형 쿼리 라우팅
            if not RAGQueryOptimizer.is_complex_query(state["input"]):
                logger.info(f"[Optimizer] 단순 질문 감지 -> 확장을 건너뜁니다: {state['input']}")
                from core.session import SessionManager
                SessionManager.replace_last_status_log("질문 분석 완료")
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
            from core.session import SessionManager
            SessionManager.replace_last_status_log("질문 정밀 분석 완료")
            SessionManager.add_status_log("문서 검색 중")
            
            # [개선] 불렛 포인트 및 번호 제거 파싱 로직
            raw_queries = [q.strip() for q in result.split("\n") if q.strip()]
            clean_queries = []
            for q in raw_queries:
                clean_q = clean_query_text(q)
                if clean_q:
                    clean_queries.append(clean_q)
            
            final_queries = clean_queries[:3] if clean_queries else [state["input"]]
            logger.info(f"[Graph] 확장된 쿼리: {final_queries} (개수: {len(final_queries)})")
            op.tokens = len(result.split())
            return {"search_queries": final_queries}

        except Exception as e:
            op.error = str(e)
            logger.warning(f"[Graph] 쿼리 확장 실패: {e}")
            return {"search_queries": [state["input"]]}


    # 2. 문서 검색 노드 (AsyncIO 최적화)
    async def retrieve_documents(state: GraphState) -> GraphOutput:
        """
        [AsyncIO 최적화] 여러 쿼리로부터 병렬 문서 검색
        - ConcurrentDocumentRetriever를 사용한 동시 검색
        - SHA256 기반 중복 제거
        - 메타데이터 통합
        """
        with monitor.track_operation(
            OperationType.DOCUMENT_RETRIEVAL, 
            {"query_count": len(state.get("search_queries", [state["input"]]))}
        ) as op:
            queries = state.get("search_queries", [state["input"]])
            logger.info(f"[Graph] 병렬 문서 검색 시작 (AsyncIO 최적화) - {len(queries)} 쿼리")
        
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
                        f"[Graph] 검색 타임아웃 ({TimeoutConstants.RETRIEVER_TIMEOUT}초 초과): {q}"
                    )
                    return []
                except Exception as e:
                    logger.error(f"[Graph] 검색 오류 ({q}): {e}", exc_info=True)
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
                    f"[Graph] 병렬 검색 완료: "
                    f"검색: {retrieval_stats['total_retrieved']}, "
                    f"중복 제거: {retrieval_stats['duplicates_removed']}, "
                    f"최종: {retrieval_stats['unique_count']}"
                )
                from core.session import SessionManager
                SessionManager.replace_last_status_log(f"관련 문장 {retrieval_stats['unique_count']}개 식별")
                SessionManager.add_status_log("핵심 문장 엄선 중")
                op.tokens = sum(len(doc.page_content.split()) for doc in unique_docs)
                return {"documents": unique_docs}
            
            except Exception as e:
                logger.error(f"[Graph] 동시 검색 최적화 실패: {e}")
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
                
                logger.info(f"[Graph] 폴백 검색 완료: {len(unique_docs)} 문서")
                op.tokens = sum(len(doc.page_content.split()) for doc in unique_docs)
                return {"documents": unique_docs}


    # 3. 재순위화 노드 (지능형 최적화)
    async def rerank_documents(state: GraphState) -> GraphOutput:
        """
        [지능형 최적화] 리랭킹 성능 및 자원 효율화
        - 점수 기반 바이패스 (Score-based Bypassing)
        - 대상 문서 수 최적화 (Adaptive Top-K)
        - 비동기 배치 처리
        """
        documents = state.get("documents", [])
        if not RERANKER_CONFIG.get("enabled", False) or not documents:
            return {"documents": documents}

        with monitor.track_operation(
            OperationType.RERANKING, 
            {"doc_count": len(documents)}
        ) as op:
            try:
                # [최적화 1] 바이패스 로직
                # - 문서가 1개뿐이거나, 이미 신뢰도가 높다고 판단되는 경우 생략
                # - 현재 검색 구조에서는 점수가 명시적이지 않으므로, 
                # - 우선은 개수가 너무 적을 때(품질이 명확할 가능성 높음) 생략하는 로직 포함
                if len(documents) <= 1:
                    logger.info("[Optimizer] 문서가 1개뿐이므로 리랭킹을 생략합니다.")
                    return {"documents": documents}

                # [최적화 2] 대상 문서 제한 (VRAM 및 속도 보호)
                max_docs = RERANKER_CONFIG.get("max_rerank_docs", 12)
                if len(documents) > max_docs:
                    logger.info(f"[Optimizer] 리랭킹 대상을 상위 {max_docs}개로 제한합니다. (원본: {len(documents)})")
                    documents = documents[:max_docs]

                logger.info(f"[Graph] 지능형 리랭킹 시작 - {len(documents)} 문서")
                reranker = load_reranker_model(RERANKER_CONFIG.get("model_name"))
                top_k = RERANKER_CONFIG.get("top_k", 6)
                
                async def _rerank_batch(query: str, docs: DocumentList) -> List[float]:
                    try:
                        pairs = [[query, doc.page_content] for doc in docs]
                        # 리랭커는 CPU에서 돌리거나 모델을 캐싱하여 VRAM 경합 방지 권장
                        scores = await asyncio.to_thread(reranker.predict, pairs)
                        return scores
                    except Exception as e:
                        logger.error(f"[Graph] 리랭킹 배치 오류: {e}")
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
                
                from core.session import SessionManager
                SessionManager.replace_last_status_log(f"핵심 문장 {rerank_stats['output_count']}개 엄선")
                SessionManager.add_status_log("답변 작성 중")

                logger.info(
                    f"[Graph] 리랭킹 완료: {rerank_stats['input_count']} -> {rerank_stats['output_count']} 문서"
                )
                op.tokens = sum(len(doc.page_content.split()) for doc in final_docs)
                return {"documents": final_docs}
            
            except Exception as e:
                op.error = str(e)
                logger.error(f"[Graph] 리랭킹 실패 (폴백 적용): {e}")
                return {"documents": documents[:6]} # 실패 시 상위 6개 반환


    # 4. 컨텍스트 포맷팅 노드
    def format_context(state: GraphState) -> GraphOutput:
        documents = state["documents"]
        if not documents:
            return {"context": ""}

        merged_docs = _merge_consecutive_chunks(documents)
        formatted = []
        for i, doc in enumerate(merged_docs):
            # [개선] 페이지 번호만 포함하여 LLM이 [p.X] 형식으로 인용하도록 강력 유도
            page = doc.metadata.get("page", "?")
            formatted.append(f"[p.{page}]\n{doc.page_content}")
        
        context_str = "\n\n".join(formatted)
        # [Debug] 컨텍스트의 앞부분만 로그로 확인 (페이지 번호 포함 여부 체크)
        logger.info(f"[Graph] 컨텍스트 생성 완료 (Sample): {context_str[:200].replace(chr(10), ' ')}...")
        return {"context": context_str}


    # 5. 답변 생성 노드 (가장 중요)
    async def generate_response(state: GraphState, config: RunnableConfig) -> GraphOutput:
        """
        LLM 답변을 생성하고 스트리밍합니다. 
        내부적으로 성능 지표(TTFT, 추론 시간 등)를 상세히 기록합니다.
        """
        with monitor.track_operation(OperationType.LLM_INFERENCE, {"doc_count": len(state.get("documents", []))}) as op:
            try:
                logger.info("[Graph] LLM 답변 생성 프로세스 시작")
                from core.session import SessionManager
                SessionManager.add_status_log("답변 작성 중")
                
                llm = config.get("configurable", {}).get("llm")
                if not llm:
                    raise ValueError("LLM 인스턴스가 설정(config)에 누락되었습니다.")

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", QA_SYSTEM_PROMPT),
                    ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
                ])

                # StrOutputParser를 쓰지 않고 AIMessageChunk를 직접 받아 메타데이터(thinking) 유지
                generation_chain = prompt_template | llm

                # 성능 측정을 위한 내부 상태 추적기
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

                    def record_chunk(self, content: str, thought: str):
                        now = time.time()
                        if self.first_token_at is None:
                            self.first_token_at = now
                        
                        if thought:
                            if self.thinking_started_at is None:
                                self.thinking_started_at = now
                            self.full_thought += thought
                            
                        if content:
                            if self.answer_started_at is None:
                                self.answer_started_at = now
                                # 답변이 시작되면 사고 과정은 끝난 것으로 간주
                                if self.thinking_started_at and self.thinking_finished_at is None:
                                    self.thinking_finished_at = now
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
                        
                        # 토큰 수(단어 단위) 및 속도 계산
                        token_count = len(self.full_response.split())
                        tokens_per_second = token_count / answer_duration if answer_duration > 0 else 0
                        
                        # 1. 단일행 로그 기록
                        logger.info(
                            f"[LLM Metrics] TTFT: {time_to_first_token:.2f}s | "
                            f"Thinking: {thinking_duration:.2f}s | "
                            f"Answer: {answer_duration:.2f}s | "
                            f"Total: {total_duration:.2f}s | "
                            f"Tokens: {token_count} | "
                            f"Speed: {tokens_per_second:.1f} tok/s"
                        )

                        # 2. CSV 파일 영구 기록
                        try:
                            monitor.log_to_csv({
                                "model": getattr(self.model, "model", "unknown"),
                                "ttft": time_to_first_token,
                                "thinking": thinking_duration,
                                "answer": answer_duration,
                                "total": total_duration,
                                "tokens": token_count,
                                "tps": tokens_per_second,
                                "query": self.query
                            })
                        except Exception as e:
                            logger.warning(f"성능 지표 CSV 저장 실패: {e}")

                tracker = ResponsePerformanceTracker(state["input"], llm)
                timeout = TimeoutConstants.LLM_TIMEOUT

                async def _consume_stream_and_dispatch_events() -> None:
                    """모델 스트림을 소비하고 가공하여 커스텀 이벤트를 발생시킵니다."""
                    async for chunk in generation_chain.astream(
                        {"input": state["input"], "context": state["context"]},
                        config=config,
                    ):
                        # 데이터 추출
                        content = getattr(chunk, "content", "")
                        thought = ""
                        if hasattr(chunk, "additional_kwargs"):
                            thought = chunk.additional_kwargs.get("thought") or chunk.additional_kwargs.get("thinking", "")
                        
                        if not content and not thought:
                            continue

                        # 메트릭 업데이트
                        tracker.record_chunk(content, thought)

                        # UI로 실시간 전송
                        await adispatch_custom_event(
                            "response_chunk",
                            {"chunk": content, "thought": thought},
                            config=config,
                        )
                    
                    tracker.finalize_and_log()

                # 스트리밍 태스크 실행 및 타임아웃 관리
                generation_task = asyncio.create_task(_consume_stream_and_dispatch_events())

                try:
                    await asyncio.wait_for(generation_task, timeout=timeout)

                except asyncio.TimeoutError:
                    logger.error(f"[Graph] LLM 응답 생성 시간 초과 ({timeout}초)")
                    generation_task.cancel()
                    try: await generation_task
                    except asyncio.CancelledError: pass
                    
                    if not tracker.full_response:
                        tracker.full_response = "죄송합니다. 응답 생성 시간이 너무 오래 걸려 중단되었습니다."
                    op.error = "timeout"
                
                except Exception as e:
                    if not generation_task.done():
                        generation_task.cancel()
                        try: await generation_task
                        except: pass
                    logger.error(f"[Graph] 응답 생성 중 예외 발생: {e}", exc_info=True)
                    op.error = str(e)
                    raise
                
                # 최종 상태 업데이트 및 반환
                op.tokens = len(tracker.full_response.split())
                from core.session import SessionManager
                SessionManager.replace_last_status_log("답변 생성 완료")
                
                return {
                    "response": tracker.full_response, 
                    "documents": state.get("documents", [])
                }
            except Exception as e:
                logger.error(f"[Graph] generate_response 노드 치명적 오류: {e}")
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
