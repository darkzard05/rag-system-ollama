"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
Pydantic 모델, TypedDict 등이 포함됩니다.
"""

from typing import List, TypedDict, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    """
    input: str
    search_queries: List[str]  # 다중 검색 쿼리 리스트
    documents: List[Document]
    context: Optional[str]
    response: Optional[str]