"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
Pydantic 모델, TypedDict 등이 포함됩니다.
"""
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# --- LangGraph의 상태 객체 ---
class GraphState(TypedDict):
    """
    그래프의 각 노드를 거치며 전달될 상태 객체입니다.
    """
    input: str
    documents: List[Document]
    context: str
    response: str