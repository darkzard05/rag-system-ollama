"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
"""

import logging
from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from config import QA_SYSTEM_PROMPT
from session import SessionManager


# --- 1. Graph State 정의 ---
class GraphState(TypedDict):
    """
    그래프의 각 노드를 거치며 전달될 상태 객체입니다.
    """
    input: str
    documents: List[Document]
    context: str
    generation: str
    retries: int = 0


# --- 2. Graph 노드 정의 ---
def retrieve_documents(state: GraphState) -> GraphState:
    """사용자 질문을 기반으로 문서를 검색합니다."""
    logging.info("노드 실행: retrieve_documents")
    user_input = state["input"]
    retriever = SessionManager.get("retriever")
    if not retriever:
        raise ValueError("세션에서 리트리버를 찾을 수 없습니다. RAG 파이프라인이 올바르게 빌드되었는지 확인하세요.")
    
    documents = retriever.invoke(user_input)
    return {"documents": documents}

def format_context(state: GraphState) -> GraphState:
    """검색된 문서를 LLM에 전달할 최종 컨텍스트로 포맷합니다."""
    logging.info("노드 실행: format_context")
    documents = state["documents"]
    
    formatted_docs = []
    for i, doc in enumerate(documents):
        doc.metadata["doc_number"] = i + 1
        page_number = doc.metadata.get("page", "N/A")
        if page_number != "N/A":
            doc.metadata["page"] = str(int(page_number) + 1)

        formatted_docs.append(
            f"[{doc.metadata['doc_number']}] {doc.page_content} (p.{doc.metadata.get('page', 'N/A')})"
        )
    
    context = "\n\n".join(formatted_docs)
    return {"context": context}

def generate_answer(state: GraphState) -> GraphState:
    """컨텍스트와 질문을 기반으로 LLM 답변을 생성합니다."""
    logging.info("노드 실행: generate_answer")
    user_input = state["input"]
    context = state["context"]
    llm = SessionManager.get("llm")
    if not llm:
        raise ValueError("세션에서 LLM을 찾을 수 없습니다. 모델이 올바르게 로드되었는지 확인하세요.")

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            (
                "human",
                "Based on the context below, answer the following question.\n\nQuestion: {input}\n\n[Context]\n{context}",
            ),
        ]
    )
    
    rag_chain = qa_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"input": user_input, "context": context})
    return {"generation": generation}


# --- 3. Graph 구성 및 컴파일 ---
def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일합니다."""
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate", generate_answer)

    # 엣지 설정 (순서 정의)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    # 그래프 컴파일
    app = workflow.compile()
    logging.info("LangGraph 워크플로우가 성공적으로 컴파일되었습니다.")
    return app
