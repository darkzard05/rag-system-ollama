"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
Core Logic Rebuild: 데코레이터 제거 및 순수 함수 구조로 변경하여 config 전달 보장.
"""

import logging
import asyncio
import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import StateGraph, END

from config import QA_SYSTEM_PROMPT, RERANKER_CONFIG, QUERY_EXPANSION_PROMPT, QUERY_EXPANSION_CONFIG
from schemas import GraphState
from model_loader import load_reranker_model
from utils import clean_query_text

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _merge_consecutive_chunks(docs: List[Document]) -> List[Document]:
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

def build_graph(retriever: Any):
    """
    RAG 워크플로우를 구성하고 컴파일합니다.
    """
    # 1. 쿼리 확장 노드
    async def generate_queries(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        logger.info("[Graph] 쿼리 생성 시작")
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
            # invoke 대신 ainvoke 사용
            result = await chain.ainvoke({"input": state["input"]})
            
            # [개선] 불렛 포인트 및 번호 제거 파싱 로직
            raw_queries = [q.strip() for q in result.split("\n") if q.strip()]
            clean_queries = []
            for q in raw_queries:
                clean_q = clean_query_text(q)
                if clean_q:
                    clean_queries.append(clean_q)
            
            final_queries = clean_queries[:3] if clean_queries else [state["input"]]
            logger.info(f"[Graph] 확장된 쿼리: {final_queries}")
            return {"search_queries": final_queries}

        except Exception as e:
            logger.warning(f"[Graph] 쿼리 확장 실패: {e}")
            return {"search_queries": [state["input"]]}


    # 2. 문서 검색 노드
    async def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        queries = state.get("search_queries", [state["input"]])
        logger.info(f"[Graph] 검색 시작 ({len(queries)} 쿼리)")
        
        async def _safe_ainvoke(q):
            try:
                if hasattr(retriever, "ainvoke"):
                    return await retriever.ainvoke(q)
                return await asyncio.to_thread(retriever.invoke, q)
            except Exception as e:
                logger.error(f"검색 오류 ({q}): {e}")
                return []

        results = await asyncio.gather(*[_safe_ainvoke(q) for q in queries])
        all_documents = [doc for sublist in results for doc in sublist]
        
        # [수정] 중복 제거: 내용 기반 해싱 (정확도 복구 및 메모리 최적화)
        unique_docs = []
        seen = set()
        
        for doc in all_documents:
            doc_hash = hash(doc.page_content)
            
            if doc_hash not in seen:
                unique_docs.append(doc)
                seen.add(doc_hash)
        
        logger.info(f"[Graph] 검색 완료: {len(unique_docs)} 문서")
        return {"documents": unique_docs}


    # 3. 재순위화 노드
    def rerank_documents(state: GraphState) -> Dict[str, Any]:
        documents = state["documents"]
        if not RERANKER_CONFIG.get("enabled", False) or not documents:
            return {"documents": documents}

        try:
            logger.info("[Graph] 재순위화 시작")
            reranker = load_reranker_model(RERANKER_CONFIG.get("model_name"))
            pairs = [[state["input"], doc.page_content] for doc in documents]
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            
            top_k = RERANKER_CONFIG.get("top_k", 5)
            final_docs = [doc for doc, score in scored_docs[:top_k]]
            return {"documents": final_docs}
        except Exception as e:
            logger.error(f"[Graph] 재순위화 실패: {e}")
            return {"documents": documents}


    # 4. 컨텍스트 포맷팅 노드
    def format_context(state: GraphState) -> Dict[str, Any]:
        documents = state["documents"]
        if not documents:
            return {"context": ""}

        merged_docs = _merge_consecutive_chunks(documents)
        formatted = []
        for i, doc in enumerate(merged_docs):
            # [개선] 페이지 번호만 포함하여 LLM이 [p.X] 형식으로 인용하도록 강력 유도
            page = doc.metadata.get("page", "?")
            formatted.append(f"[p.{page}]\n{doc.page_content}")
            
        return {"context": "\n\n".join(formatted)}


    # 5. 답변 생성 노드 (가장 중요)
    async def generate_response(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        config가 확실하게 전달되도록 데코레이터 없이 구현함.
        chain.astream을 직접 순회하여 토큰 생성 이벤트를 강제로 발생시킴.
        """
        logger.info("[Graph] 답변 생성 시작 (Streaming 실행)")
        llm = config.get("configurable", {}).get("llm")
        if not llm:
            raise ValueError("LLM is missing in config")

        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "[Context]:\n{context}\n\n[Question]: {input}"),
        ])

        chain = prompt | llm | StrOutputParser()

        full_response = ""
        # ainvoke 대신 astream을 사용하여 토큰 단위로 받으며, config를 통해 이벤트를 전파함
        async for chunk in chain.astream(
            {"input": state["input"], "context": state["context"]},
            config=config
        ):
            full_response += chunk
            # [Fix] 강제 스트리밍: 표준 콜백이 버퍼링될 경우를 대비해 커스텀 이벤트 발송
            await adispatch_custom_event("response_chunk", {"chunk": chunk}, config=config)
        
        # UI에서 출처 표시를 할 수 있도록 문서 목록도 함께 유지
        return {"response": full_response, "documents": state.get("documents", [])}


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
