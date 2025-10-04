"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
"""
import logging
from typing import Any, AsyncGenerator, Dict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from config import QA_SYSTEM_PROMPT
from session import SessionManager
from utils import async_log_operation
from schemas import GraphState


# 💡 build_graph가 더 이상 llm을 인자로 받지 않도록 수정
def build_graph(retriever: Any):
    """
    단순한 RAG 워크플로우를 구성하고 컴파일합니다.
    """

    # 라우터 및 분기 노드(conversational, general) 함수들 모두 삭제

    def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        logging.info("노드 실행: retrieve_documents")
        if not retriever:
            raise ValueError("그래프에 유효한 리트리버가 주입되지 않았습니다.")
        documents = retriever.invoke(state["input"])
        return {"documents": documents}

    def format_context(state: GraphState) -> Dict[str, Any]:
        logging.info("노드 실행: format_context")
        documents = state["documents"]
        logging.info(f"리트리버가 {len(documents)}개의 문서를 검색했습니다.")
        if not documents:
            logging.warning("검색된 문서가 없어 컨텍스트가 비어있습니다.")
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
        logging.info(f"생성된 컨텍스트의 길이: {len(context)}자")
        return {"context": context}

    @async_log_operation("통합 응답 생성")
    async def generate_response(state: GraphState) -> AsyncGenerator[Dict[str, Any], None]:
        # 💡 llm을 SessionManager에서 다시 가져오도록 수정
        llm = SessionManager.get("llm")
        if not llm:
            raise ValueError("세션에서 LLM을 찾을 수 없습니다.")
            
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", "[Context]:\n{context}\n\n[Question]: {input}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        async for chunk in chain.astream({"input": state["input"], "context": state["context"]}):
            yield {"response": chunk}

    # --- 💡 그래프 연결 로직을 매우 단순하게 변경 💡 ---
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    app = workflow.compile()
    logging.info("단순 RAG LangGraph 워크플로우가 성공적으로 컴파일되었습니다.")
    
    return app