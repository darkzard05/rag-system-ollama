"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Core Logic Rebuild: 데코레이터 제거 및 순수 함수 구조로 변경하여 config 전달 보장.
"""

import asyncio
import contextlib
import hashlib
import logging
import time
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from api.schemas import GraphState
from common.config import (
    QA_HUMAN_PROMPT,
    QA_SYSTEM_PROMPT,
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
from common.utils import clean_query_text
from core.model_loader import load_reranker_model
from core.query_optimizer import RAGQueryOptimizer  # 추가
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization.async_optimizer import get_concurrent_document_reranker

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

        # 같은 소스, 같은 페이지, 연속된 인덱스일 때만 병합
        if (
            curr_src == next_src
            and curr_page == next_page
            and curr_idx != -1
            and next_idx == curr_idx + 1
        ):
            current_doc.page_content += " " + next_doc.page_content
            # 인덱스 업데이트 (마지막 인덱스로)
            current_doc.metadata["chunk_index"] = next_idx
        else:
            merged.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content, metadata=next_doc.metadata.copy()
            )
    merged.append(current_doc)
    return merged


# --- Graph Construction ---


def build_graph(retriever: T | None = None) -> T:
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

    # --- New Nodes ---

    async def intent_router(state: GraphState, config: RunnableConfig) -> GraphOutput:
        """질문의 의도를 분석하여 경로를 결정하고 기본 상태를 초기화합니다."""
        llm = config.get("configurable", {}).get("llm")
        if not llm:
            return {"route_decision": "FACTOID", "search_queries": [state["input"]]}

        intent = await RAGQueryOptimizer.classify_intent(state["input"], llm)

        # 기본 출력 구성 (상태 오염 방지)
        output: GraphOutput = {
            "route_decision": intent,
            "search_queries": [state["input"]],
            "documents": [],
            "context": "",
        }

        # 특정 의도에 따른 사전 설정
        if intent == "GREETING":
            output["context"] = (
                "(시스템: 이 질문은 일상적인 대화입니다. 문서 참조 없이 친절하게 답하세요.)"
            )

        return output

    # 1. 쿼리 확장 노드 (AsyncIO 최적화)
    async def generate_queries(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        """
        [AsyncIO 최적화] 쿼리 확장 시 동시 처리
        """
        from core.session import SessionManager

        # ... (이전 로직 유지)
        with monitor.track_operation(
            OperationType.QUERY_PROCESSING, {"stage": "async_query_expansion"}
        ) as op:
            logger.info(
                f"[Query] [Start] 질문 분석 및 검색어 확장 시작: '{state['input'][:50]}...'"
            )

            # [최적화] 의도 기반 지능형 라우팅 적용
            # 이미 'router' 노드에서 RESEARCH(심층 분석)로 판단된 경우에만 이 노드에 진입하므로,
            # 별도의 글자 수 체크 없이 즉시 확장을 수행합니다.
            if not QUERY_EXPANSION_CONFIG.get("enabled", True):
                logger.info("[Query] [Skip] 쿼리 확장 비활성화 상태")
                return {"search_queries": [state["input"]]}

        llm = config.get("configurable", {}).get("llm")
        if not llm:
            return {"search_queries": [state["input"]]}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_EXPANSION_PROMPT),
                ("human", "{input}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()

        try:
            # [최적화] 비동기 LLM 호출
            result = await chain.ainvoke({"input": state["input"]})

            # [개선] 쿼리 정규화 및 필터링 강화
            raw_queries = [q.strip() for q in result.split("\n") if q.strip()]
            clean_queries = []
            seen = set()

            # 원본 질문을 우선적으로 포함 (선택적: 품질 보장용)
            # clean_queries.append(state["input"])
            # seen.add(state["input"])

            for q in raw_queries:
                clean_q = clean_query_text(q)
                # 너무 짧거나(2자 미만) 너무 긴(100자 초과) 비정상 쿼리 필터링
                if clean_q and 2 <= len(clean_q) <= 100 and clean_q not in seen:
                    clean_queries.append(clean_q)
                    seen.add(clean_q)

            # 최대 3개까지만 사용하고, 유효한 결과가 없으면 원본 질문 사용
            final_queries = clean_queries[:3] if clean_queries else [state["input"]]

            logger.info(
                f"[Query] [Complete] 검색어 {len(final_queries)}개 생성 완료: {final_queries}"
            )
            SessionManager.replace_last_status_log(
                f"질문 분석 완료 (검색어 {len(final_queries)}개)"
            )
            SessionManager.add_status_log("문서 검색 중...")

            op.tokens = len(result.split())
            return {"search_queries": final_queries}

        except Exception as e:
            op.error = str(e)
            logger.warning(f"[Query] [Error] 쿼리 확장 실패 (폴백 적용): {e}")
            return {"search_queries": [state["input"]]}

    # 2. 문서 검색 노드 (AsyncIO 최적화)
    async def retrieve_documents(state: GraphState) -> GraphOutput:
        """
        [AsyncIO 최적화] 여러 쿼리 및 다중 리트리버로부터 병렬 문서 검색
        """
        from core.session import SessionManager

        with monitor.track_operation(
            OperationType.DOCUMENT_RETRIEVAL,
            {"query_count": len(state.get("search_queries", [state["input"]]))},
        ) as op:
            queries = state.get("search_queries", [state["input"]])
            logger.info(
                f"[Retrieval] [Start] 병렬 하이브리드 검색 시작 ({len(queries)}개 검색어)"
            )

            # 세션에서 개별 리트리버 획득
            bm25 = SessionManager.get("bm25_retriever")
            faiss = SessionManager.get("faiss_retriever")

            async def _invoke_retriever(ret, q: str) -> DocumentList:
                """리트리버 비동기 호출 헬퍼"""
                if not ret:
                    return []
                try:
                    if hasattr(ret, "ainvoke"):
                        return await asyncio.wait_for(
                            ret.ainvoke(q), timeout=TimeoutConstants.RETRIEVER_TIMEOUT
                        )
                    return await asyncio.wait_for(
                        asyncio.to_thread(ret.invoke, q),
                        timeout=TimeoutConstants.RETRIEVER_TIMEOUT,
                    )
                except Exception as e:
                    logger.warning(f"[Retrieval] 검색 실패 ({type(ret).__name__}): {e}")
                    return []

            # 모든 쿼리에 대해 BM25와 FAISS를 병렬로 실행
            tasks = []
            for q in queries:
                tasks.append(_invoke_retriever(bm25, q))
                tasks.append(_invoke_retriever(faiss, q))

            # [최적화 핵심] asyncio.gather로 모든 검색 작업을 한 번에 수행
            results = await asyncio.gather(*tasks)
            all_documents = [doc for sublist in results for doc in sublist]

            # 중복 제거 (SHA256 해시 기반)
            unique_docs = []
            seen = set()
            for doc in all_documents:
                doc_key = doc.page_content + doc.metadata.get("source", "")
                doc_hash = hashlib.sha256(doc_key.encode()).hexdigest()
                if doc_hash not in seen:
                    unique_docs.append(doc)
                    seen.add(doc_hash)

            logger.info(
                f"[Retrieval] [Complete] 병렬 하이브리드 검색 완료: {len(unique_docs)} 문서 확보"
            )
            SessionManager.replace_last_status_log(
                f"문서 {len(unique_docs)}개 검색 완료"
            )
            SessionManager.add_status_log("핵심 문장 선별 중...")

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
            OperationType.RERANKING, {"doc_count": len(documents)}
        ) as op:
            try:
                # [최적화 1] 바이패스 로직
                if len(documents) <= 1:
                    logger.info("[Rerank] [Skip] 문서가 1개뿐이므로 생략")
                    return {"documents": documents}

                # [최적화 2] 대상 문서 제한 (VRAM 및 속도 보호)
                max_docs = RERANKER_CONFIG.get("max_rerank_docs", 12)
                if len(documents) > max_docs:
                    logger.info(
                        f"[Rerank] [Limit] 대상 제한: {len(documents)} -> {max_docs}"
                    )
                    documents = documents[:max_docs]

                logger.info(
                    f"[Rerank] [Start] 지능형 리랭킹 시작 ({len(documents)}개 문서)"
                )

                try:
                    reranker = load_reranker_model(RERANKER_CONFIG.get("model_name"))
                    if not reranker:
                        raise ValueError("리랭커 모델을 로드할 수 없습니다.")
                except Exception as e:
                    logger.warning(
                        f"[Rerank] [Skip] 리랭커 로드 실패, 원본 순위 유지: {e}"
                    )
                    return {"documents": documents[: RERANKER_CONFIG.get("top_k", 6)]}

                top_k = RERANKER_CONFIG.get("top_k", 6)

                async def _rerank_batch(query: str, docs: DocumentList) -> list[float]:
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
                    metadata={"pipeline": "optimized_concurrent"},
                )

                logger.info(
                    f"[Rerank] [Complete] 리랭킹 완료: {rerank_stats['input_count']} -> {rerank_stats['output_count']} 문서 선정"
                )
                SessionManager.replace_last_status_log(
                    f"핵심 문장 {rerank_stats['output_count']}개 선별 완료"
                )
                SessionManager.add_status_log("답변 생성 중...")

                op.tokens = sum(len(doc.page_content.split()) for doc in final_docs)
                return {"documents": final_docs}

            except Exception as e:
                op.error = str(e)
                logger.error(f"[Rerank] [Error] 리랭킹 실패 (폴백 적용): {e}")
                return {"documents": documents[:6]}  # 실패 시 상위 6개 반환

    # 4. 컨텍스트 포맷팅 노드
    def format_context(state: GraphState) -> GraphOutput:
        documents = state.get("documents", [])
        if not documents:
            return {"context": ""}

        # [최적화] Ollama 프롬프트 캐싱(KV Cache)을 위해 문서들을 일정한 순서로 정렬
        # 출처(source), 페이지(page), 청크 인덱스(chunk_index) 순으로 정렬하여
        # 검색 결과의 순서가 바뀌더라도 프롬프트의 일관성을 유지합니다.
        sorted_docs = sorted(
            documents,
            key=lambda d: (
                d.metadata.get("source", ""),
                d.metadata.get("page", 0),
                d.metadata.get("chunk_index", -1),
            ),
        )

        merged_docs = _merge_consecutive_chunks(sorted_docs)
        formatted = []
        for _i, doc in enumerate(merged_docs):
            page = doc.metadata.get("page", "?")
            # 모델이 출력 형식과 혼동하지 않도록 형식을 변경합니다.
            formatted.append(
                f"-- DOCUMENT CONTENT (PAGE {page}) --\n{doc.page_content}"
            )

        context_str = "\n\n".join(formatted)
        return {"context": context_str}

    # 5. 답변 생성 노드 (가장 중요)
    async def generate_response(
        state: GraphState, config: RunnableConfig
    ) -> GraphOutput:
        """
        LLM 답변을 생성하고 스트리밍합니다.
        내부적으로 성능 지표(TTFT, 추론 시간 등)를 상세히 기록합니다.
        """
        from common.utils import get_ollama_resource_usage
        from core.session import SessionManager

        # [수정] documents가 None이거나 없을 경우를 대비한 안전한 카운팅
        docs = state.get("documents") or []
        doc_count = len(docs)

        with monitor.track_operation(
            OperationType.LLM_INFERENCE, {"doc_count": doc_count}
        ) as op:
            try:
                llm = config.get("configurable", {}).get("llm")
                if not llm:
                    raise ValueError("LLM 인스턴스가 설정(config)에 누락되었습니다.")

                model_name = getattr(llm, "model", "Unknown")
                resource_status = get_ollama_resource_usage(model_name)

                # [최적화] 리소스 상세 정보는 백엔드 로그로만 기록 (UI 단순화)
                logger.info(
                    f"[LLM] [Start] 답변 생성 프로세스 시작 (Model: {model_name}, Resource: {resource_status})"
                )
                SessionManager.add_status_log("답변 생성 시작")

                # [리팩토링] 의도에 따라 시스템 및 휴먼 프롬프트 동적 변경
                intent = state.get("route_decision", "FACTOID")

                if intent == "GREETING":
                    sys_prompt = "당신은 친절한 AI 어시스턴트입니다. 사용자의 인사나 일상적인 대화에 자연스럽고 따뜻하게 응답하세요."
                    human_prompt = "{input}"  # 문서 컨텍스트 참조 지시 제거
                else:
                    sys_prompt = QA_SYSTEM_PROMPT
                    human_prompt = QA_HUMAN_PROMPT

                ChatPromptTemplate.from_messages(
                    [
                        ("system", sys_prompt),
                        ("human", human_prompt),
                    ]
                )

                # 1. 최적화: 메시지 사전 포맷팅 (Ollama SDK 직접 호출용)
                formatted_messages = [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": human_prompt.format(
                            input=state["input"], context=state.get("context", "")
                        ),
                    },
                ]

                # 2. 최적화: Ollama 옵션 사전 설정
                ollama_options = {
                    "temperature": getattr(llm, "temperature", 0.5),
                    "num_predict": getattr(llm, "num_predict", -1),
                    "top_p": getattr(llm, "top_p", 0.9),
                    "num_ctx": getattr(llm, "num_ctx", 4096),
                }

                class ResponsePerformanceTracker:
                    def __init__(self, query_text: str, model_instance: Any):
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
                                # 최초 1회만 수행
                                SessionManager.replace_last_status_log(
                                    "사고 과정 기록 중..."
                                )
                                self.thinking_started_at = now
                                self._log_thinking_start = True
                            self._thought_parts.append(thought)

                        if content:
                            if not self._log_answer_start:
                                # 최초 1회만 수행
                                if (
                                    self.thinking_started_at
                                    and self.thinking_finished_at is None
                                ):
                                    self.thinking_finished_at = now

                                SessionManager.replace_last_status_log(
                                    "답변 스트리밍 중..."
                                )
                                self.answer_started_at = now
                                self._log_answer_start = True

                            self._resp_parts.append(content)
                            self.chunk_count += 1

                    def finalize_and_log(self):
                        self.answer_finished_at = time.time()
                        self.full_response = "".join(self._resp_parts)
                        self.full_thought = "".join(self._thought_parts)

                        total_duration = self.answer_finished_at - self.start_time
                        time_to_first_token = (
                            (self.first_token_at - self.start_time)
                            if self.first_token_at
                            else 0
                        )

                        thinking_duration = 0
                        if self.thinking_started_at:
                            end_time = (
                                self.thinking_finished_at
                                or self.answer_started_at
                                or self.answer_finished_at
                            )
                            thinking_duration = end_time - self.thinking_started_at

                        answer_duration = (
                            (self.answer_finished_at - self.answer_started_at)
                            if self.answer_started_at
                            else 0
                        )
                        resp_token_count = len(self.full_response.split())
                        thought_token_count = len(self.full_thought.split())
                        tokens_per_second = (
                            (resp_token_count / answer_duration)
                            if answer_duration > 0
                            else 0
                        )

                        logger.info(
                            f"[LLM] [Complete] 생성 완료 | "
                            f"TTFT: {time_to_first_token:.2f}s | "
                            f"사고: {thinking_duration:.2f}s ({thought_token_count} tok) | "
                            f"답변: {answer_duration:.2f}s ({resp_token_count} tok) | "
                            f"속도: {tokens_per_second:.1f} tok/s | "
                            f"총 소요: {total_duration:.2f}s"
                        )

                        SessionManager.replace_last_status_log(
                            f"완료 (사고 {thought_token_count} / 답변 {resp_token_count})"
                        )

                        with contextlib.suppress(Exception):
                            monitor.log_to_csv(
                                {
                                    "model": getattr(self.model, "model", "unknown"),
                                    "ttft": time_to_first_token,
                                    "thinking": thinking_duration,
                                    "answer": answer_duration,
                                    "total": total_duration,
                                    "tokens": resp_token_count,
                                    "thought_tokens": thought_token_count,
                                    "tps": tokens_per_second,
                                }
                            )
                        return resp_token_count, thought_token_count

                tracker = ResponsePerformanceTracker(state["input"], llm)
                timeout = TimeoutConstants.LLM_TIMEOUT

                async def _consume_stream_and_dispatch_events() -> None:
                    """Ollama SDK를 직접 호출하여 최고 속도로 이벤트를 발생시킵니다. (최적화된 5토큰 버퍼링 적용)"""
                    try:
                        async_client = getattr(llm, "async_client", None)
                        if not async_client:
                            import ollama

                            async_client = ollama.AsyncClient(
                                host=getattr(llm, "base_url", "http://127.0.0.1:11434")
                            )

                        buffer = []
                        # 원격 저장소에서 검증된 최적의 배치 크기
                        batch_size = 5

                        async for part in await async_client.chat(
                            model=getattr(llm, "model", "unknown"),
                            messages=formatted_messages,
                            stream=True,
                            options=ollama_options,
                        ):
                            msg_part = part.get("message", {})
                            content = msg_part.get("content", "")
                            thought = msg_part.get("thinking", "")

                            # 1. 사고 과정은 즉시 전송 (반응성 유지)
                            if thought:
                                tracker.record_chunk("", thought)
                                await adispatch_custom_event(
                                    "response_chunk",
                                    {"chunk": "", "thought": thought},
                                    config=config,
                                )

                            # 2. 답변 본문은 5개 단위로 묶어서 발송 (오버헤드 최소화)
                            if content:
                                buffer.append(content)
                                if len(buffer) >= batch_size:
                                    merged_content = "".join(buffer)
                                    tracker.record_chunk(merged_content, "")
                                    await adispatch_custom_event(
                                        "response_chunk",
                                        {"chunk": merged_content, "thought": ""},
                                        config=config,
                                    )
                                    buffer = []

                        # 3. 루프 종료 후 남은 버퍼 플러시
                        if buffer:
                            merged_content = "".join(buffer)
                            tracker.record_chunk(merged_content, "")
                            await adispatch_custom_event(
                                "response_chunk",
                                {"chunk": merged_content, "thought": ""},
                                config=config,
                            )

                    except Exception as e:
                        logger.error(f"[Stream] SDK 직접 호출 중 오류: {e}")

                # 스트리밍 태스크 실행
                generation_task = asyncio.create_task(
                    _consume_stream_and_dispatch_events()
                )

                try:
                    # 지정된 타임아웃 동안 대기
                    await asyncio.wait_for(generation_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"[LLM] [Error] 응답 생성 시간 초과 ({timeout}초)")
                    op.error = "timeout"
                except asyncio.CancelledError:
                    logger.warning(
                        "[LLM] [Cancel] 사용자에 의해 또는 시스템에 의해 작업이 취소되었습니다."
                    )
                    op.error = "cancelled"
                    raise  # 상위로 취소 신호 전파
                except Exception as e:
                    logger.error(
                        f"[LLM] [Error] 응답 생성 중 예외 발생: {e}", exc_info=True
                    )
                    op.error = str(e)
                    raise
                finally:
                    # [최적화 핵심] 노드가 종료될 때 생성 태스크가 살아있다면 반드시 취소
                    if not generation_task.done():
                        generation_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            # 취소가 완료될 때까지 대기하여 자원 정리 보장
                            await generation_task

                # 최종 지표 기록 (정상 종료 시에만 실행됨)
                resp_tokens, thought_tokens = tracker.finalize_and_log()
                op.tokens = resp_tokens
                op.metadata["thought_tokens"] = thought_tokens

                return {
                    "response": tracker.full_response,
                    "thought": tracker.full_thought,
                    "documents": state.get("documents", []),
                }
            except Exception as e:
                logger.error(
                    f"[Graph] [Critical] generate_response 노드 치명적 오류: {e}"
                )
                raise

    # --- Workflow Definition ---
    workflow = StateGraph(GraphState)

    workflow.add_node("router", intent_router)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("router")

    # 조건부 라우팅 로직
    def route_decision_func(state: GraphState):
        decision = state.get("route_decision", "FACTOID")
        if decision == "GREETING":
            return "generate_response"  # 인사면 검색 건너뛰고 답변으로
        elif decision == "FACTOID":
            return "retrieve"  # 단순 사실이면 확장 건너뛰고 검색으로
        return "generate_queries"  # 복합 질문이면 확장 수행

    workflow.add_conditional_edges(
        "router",
        route_decision_func,
        {
            "generate_response": "generate_response",
            "retrieve": "retrieve",
            "generate_queries": "generate_queries",
        },
    )

    workflow.add_edge("generate_queries", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
