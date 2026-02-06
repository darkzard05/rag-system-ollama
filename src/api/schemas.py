"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class PerformanceStats(BaseModel):
    """통합 성능 메트릭 스키마"""

    ttft: float = 0.0
    thinking_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    thought_token_count: int = 0
    tps: float = 0.0
    model_name: str = "unknown"
    doc_count: int = 0


class ChatMessage(BaseModel):
    """채팅 메시지 통합 모델"""

    role: str  # user, assistant, system
    content: str
    msg_type: str = "general"  # answer, log, greeting
    thought: str | None = None
    doc_ids: list[str] = []
    metrics: dict[str, Any] | None = None
    processed_content: str | None = None
    timestamp: float = Field(default_factory=time.time)


class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    LangGraph 호환성을 위해 TypedDict 형식을 유지합니다.
    """

    input: str
    search_queries: list[str]
    route_decision: str | None  # [추가] 의도 분석 결과
    documents: list[Document]
    relevant_docs: list[Document]  # [추가] 관련성 채점을 통과한 문서들
    context: str | None
    response: str | None


class QueryRequest(BaseModel):
    """질의 요청 스키마"""

    query: str = Field(..., examples=["DeepSeek-R1의 성능은 어때?"])
    session_id: str = Field(default="default", examples=["user-123"])
    model_name: str | None = Field(
        default=None, description="사용할 LLM 모델명 (생략 시 세션 기본값 사용)"
    )
    embedding_model: str | None = Field(
        default=None, description="사용할 임베딩 모델명"
    )
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
