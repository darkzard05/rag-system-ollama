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

    from config import QA_SYSTEM_PROMPT
    from utils import async_log_operation
    from schemas import GraphState

    def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        """
        문서를 검색하는 노드 함수.
        """
        if not retriever:
            raise ValueError("A valid retriever was not provided to the graph.")
        documents = retriever.invoke(state["input"])
        return {"documents": documents}

    def format_context(state: GraphState) -> Dict[str, Any]:
        """
        검색된 문서들을 포맷팅하여 컨텍스트를 생성하는 노드 함수.
        """
        documents = state["documents"]

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
        return {"context": context}

    @async_log_operation("Generate response")
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
