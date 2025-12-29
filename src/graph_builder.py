"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
"""

import logging
from typing import Any, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


def build_graph(retriever: Any):
    """
    단순한 RAG 워크플로우를 구성하고 컴파일합니다.

    Args:
        retriever: 문서 검색에 사용할 리트리버 객체.

    Returns:
        컴파일된 RAG 워크플로우 그래프.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import StateGraph, END

    from config import QA_SYSTEM_PROMPT, RERANKER_CONFIG, QUERY_EXPANSION_PROMPT, QUERY_EXPANSION_CONFIG
    from utils import async_log_operation, log_operation
    from schemas import GraphState
    from model_loader import load_reranker_model

    async def generate_queries(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        사용자 질문을 기반으로 다중 검색 쿼리를 생성하는 노드.
        """
        # 시작 로그 수동 기록
        logger.info("작업 시작: '검색 쿼리 생성'")
        
        # 쿼리 확장 기능 비활성화 확인
        if not QUERY_EXPANSION_CONFIG.get("enabled", True):
            logger.info("쿼리 확장 기능 비활성화됨 (설정).")
            return {"search_queries": [state["input"]]}
        
        llm = config.get("configurable", {}).get("llm")
        if not llm:
            # LLM이 없으면 원본 질문만 사용
            return {"search_queries": [state["input"]]}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_EXPANSION_PROMPT),
                ("human", "{input}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        
        try:
            # LLM 호출
            result = await chain.ainvoke({"input": state["input"]})
            # 줄바꿈으로 분리하고 빈 줄 제거
            queries = [q.strip() for q in result.split("\n") if q.strip()]
            
            # ✅ 로그에 확장된 쿼리들을 명확히 표시
            logger.info("--- 확장된 검색 쿼리 ---")
            for i, q in enumerate(queries):
                logger.info(f"쿼리 {i+1}: {q}")
            logger.info("-------------------------------")
            
            return {"search_queries": queries}
        except Exception as e:
            logger.error(f"쿼리 생성 실패: {e}")
            return {"search_queries": [state["input"]]}

    async def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        """
        문서를 검색하는 노드 함수. 다중 쿼리를 병렬로 처리합니다.
        """
        import asyncio

        if not retriever:
            raise ValueError("A valid retriever was not provided to the graph.")
        
        queries = state.get("search_queries", [state["input"]])
        all_documents = []
        
        logger.info(f"검색 시작: {len(queries)}개 쿼리 병렬 수행...")
        
        # 병렬 검색 수행 헬퍼 함수
        async def _invoke_retriever(q):
            try:
                return await asyncio.to_thread(retriever.invoke, q)
            except Exception as e:
                logger.error(f"Error retrieving for query '{q}': {e}")
                return []

        # asyncio.gather로 모든 쿼리 동시 실행
        results = await asyncio.gather(*[_invoke_retriever(q) for q in queries])
        
        for docs in results:
            all_documents.extend(docs)

        # 중복 제거 (page_content 기준)
        unique_docs = []
        seen_content = set()
        for doc in all_documents:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        logger.info(f"검색 완료: 총 {len(unique_docs)}개 문서 확보 (중복 제거됨).")
        return {"documents": unique_docs}

    @log_operation("문서 재순위화")
    def rerank_documents(state: GraphState) -> Dict[str, Any]:
        """
        검색된 문서를 재순위화(Reranking)하여 정확도를 높이는 노드 함수.
        """
        documents = state["documents"]
        query = state["input"]

        if not RERANKER_CONFIG.get("enabled", False) or not documents:
            return {"documents": documents}

        try:
            reranker = load_reranker_model(RERANKER_CONFIG.get("model_name"))
            
            # (query, doc_text) 쌍 생성
            pairs = [[query, doc.page_content] for doc in documents]
            
            # 점수 계산
            scores = reranker.predict(pairs)
            
            # 점수와 문서를 튜플로 묶어서 정렬
            scored_docs = sorted(
                zip(documents, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 상위 K개 추출
            top_k = RERANKER_CONFIG.get("top_k", 5)
            final_docs = [doc for doc, score in scored_docs[:top_k]]
            
            logger.info(f"재순위화 완료: 후보 {len(documents)}개 중 상위 {len(final_docs)}개 선택.")
            return {"documents": final_docs}
            
        except Exception as e:
            logger.error(f"재순위화 실패: {e}. 원본 검색 결과 사용.")
            return {"documents": documents}

    def format_context(state: GraphState) -> Dict[str, Any]:
        """
        검색된 문서들을 포맷팅하여 컨텍스트를 생성하는 노드 함수.
        인접한 청크(순서대로 연결된)를 병합하여 문맥 단절을 방지합니다.
        """
        from langchain_core.documents import Document

        documents = state["documents"]

        if not documents:
            logger.warning("검색된 문서가 없습니다. 컨텍스트가 비어있습니다.")
            return {"context": ""}

        # 1. 정렬: 출처(source)와 청크 인덱스(chunk_index) 기준으로 정렬
        # chunk_index가 없는 경우(구버전 데이터 등)를 대비해 기본값 처리
        def get_sort_key(doc):
            return (
                doc.metadata.get("source", ""),
                doc.metadata.get("chunk_index", -1)
            )
        
        # 원본 검색 순위(유사도 순)를 유지하는 것이 좋을 수도 있지만,
        # 문맥 복원이 목표이므로 '읽기 순서'로 재정렬합니다.
        sorted_docs = sorted(documents, key=get_sort_key)

        merged_docs = []
        if sorted_docs:
            # ✅ 원본 객체 오염 방지를 위해 첫 문서의 사본을 생성하여 시작
            first_doc = sorted_docs[0]
            current_doc = Document(
                page_content=first_doc.page_content, 
                metadata=first_doc.metadata.copy()
            )
            
            for i in range(1, len(sorted_docs)):
                next_doc = sorted_docs[i]
                
                # 2. 병합 조건: 같은 파일 & 연속된 인덱스
                curr_idx = current_doc.metadata.get("chunk_index", -1)
                next_idx = next_doc.metadata.get("chunk_index", -1)
                
                if (
                    current_doc.metadata.get("source") == next_doc.metadata.get("source")
                    and curr_idx != -1 
                    and next_idx == curr_idx + 1
                ):
                    # ✅ 내용 합치기 (새로운 객체 상태 내에서만 변경됨)
                    current_doc.page_content += " " + next_doc.page_content
                    # 메타데이터 업데이트 (페이지 범위 등은 복잡해지므로 단순 유지)
                else:
                    merged_docs.append(current_doc)
                    # ✅ 다음 문서를 처리할 때도 새로운 객체로 생성
                    current_doc = Document(
                        page_content=next_doc.page_content, 
                        metadata=next_doc.metadata.copy()
                    )
            
            merged_docs.append(current_doc)

        formatted_docs = []
        for i, doc in enumerate(merged_docs):
            doc.metadata["doc_number"] = i + 1
            page_number = doc.metadata.get("page", "N/A")
            if page_number != "N/A":
                try:
                    # 단일 페이지 번호 (정수형 또는 정수 문자열)
                    page_display = str(int(page_number) + 1)
                except ValueError:
                    # 페이지 범위 (예: "0-1") 또는 기타 형식
                    if isinstance(page_number, str) and "-" in page_number:
                        try:
                            start, end = map(int, page_number.split("-"))
                            page_display = f"{start + 1}-{end + 1}"
                        except ValueError:
                            page_display = str(page_number)
                    else:
                        page_display = str(page_number)
            else:
                page_display = "N/A"
                
            formatted_docs.append(
                f"[{doc.metadata['doc_number']}] {doc.page_content} (p.{page_display})"
            )
            
        context = "\n\n".join(formatted_docs)
        return {"context": context}

    @async_log_operation("답변 생성")
    async def generate_response(
        state: GraphState,
        config: RunnableConfig
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        LLM을 사용하여 답변을 생성하는 노드 함수 (스트리밍 지원).

        LLM 객체는 config['configurable']['llm']을 통해 주입됩니다.
        """
        llm = config.get("configurable", {}).get("llm")

        if not llm:
            raise ValueError(
                "LLM object not found in config. "
                "Make sure to pass 'config={'configurable': {'llm': llm_instance}}' when invoking the graph."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", "[참고 문서]:\n{context}\n\n[질문]: {input}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()

        async for chunk in chain.astream(
            {"input": state["input"], "context": state["context"]},
            config=config
        ):
            yield {"response": chunk}

    workflow = StateGraph(GraphState)

    workflow.add_node("generate_queries", generate_queries)  # 쿼리 생성 노드 추가
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("generate_queries")  # 시작점 변경
    workflow.add_edge("generate_queries", "retrieve")  # 쿼리 생성 -> 검색
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    app = workflow.compile()
    logger.info("RAG 워크플로우 컴파일 완료.")

    return app
