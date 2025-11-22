"""
LangGraph를 사용하여 RAG 파이프라인을 구성하고 실행하는 로직을 담당합니다.
"""

import logging
from typing import Any, AsyncGenerator, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from config import QA_SYSTEM_PROMPT
from session import SessionManager
from utils import async_log_operation
from schemas import GraphState


logger = logging.getLogger(__name__)


# 💡 build_graph가 더 이상 llm을 인자로 받지 않도록 수정
def build_graph(retriever: Any):
    """
    단순한 RAG 워크플로우를 구성하고 컴파일합니다.
    """

    # 라우터 및 분기 노드(conversational, general) 함수들 모두 삭제

    def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        logger.info("Node execution: 'retrieve_documents'")
        if not retriever:
            raise ValueError("A valid retriever was not provided to the graph.")
        documents = retriever.invoke(state["input"])
        return {"documents": documents}

    def format_context(state: GraphState) -> Dict[str, Any]:
        logger.info("Node execution: 'format_context'")
        documents = state["documents"]
        logger.info(f"Retrieved {len(documents)} documents.")
        if not documents:
            logger.warning("No documents were retrieved, context will be empty.")
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
        logger.info(f"Formatted context length: {len(context)} characters")
        return {"context": context}

    @async_log_operation("Generate response")
    async def generate_response(
        state: GraphState,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # 💡 llm을 SessionManager에서 다시 가져오도록 수정
        llm = SessionManager.get("llm")
        if not llm:
            raise ValueError("LLM not found in session.")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", "[참고 문서]:\n{context}\n\n[질문]: {input}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        async for chunk in chain.astream(
            {"input": state["input"], "context": state["context"]}
        ):
            yield {"response": chunk}

    # 워크플로우 구성
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END)

    app = workflow.compile()
    logger.info("Simple RAG LangGraph workflow compiled successfully.")

    return app
