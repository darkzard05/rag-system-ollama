"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

from typing import Any, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    LangGraph 호환성을 위해 TypedDict 형식을 유지합니다.
    """

    input: str
    search_queries: list[str]
    documents: list[Document]
    context: str | None
    response: str | None
    route_decision: str | None  # 추가됨


class QueryRequest(BaseModel):
    """질의 요청 스키마"""

    query: str = Field(..., example="DeepSeek-R1의 성능은 어때?")
    session_id: str = Field(default="default", example="user-123")
    use_cache: bool = True


class QueryResponse(BaseModel):
    """질의 응답 스키마"""

    answer: str
    sources: list[dict[str, Any]] = []
    execution_time_ms: float


class SearchResult(BaseModel):
    """검색 결과 스키마"""

    content: str
    metadata: dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    """검색 응답 스키마"""

    query: str
    results: list[SearchResult]
    count: int
