"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

import time
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field, computed_field


class PerformanceStats(BaseModel):
    """통합 성능 메트릭 스키마 (Pydantic v2 최적화)"""

    ttft: float = 0.0
    thinking_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    thought_token_count: int = 0
    model_name: str = "unknown"
    doc_count: int = 0

    @computed_field
    @property
    def tps(self) -> float:
        """Tokens Per Second (계산된 필드)"""
        return (
            self.token_count / self.generation_time if self.generation_time > 0 else 0.0
        )


class ChatMessage(BaseModel):
    """채팅 메시지 통합 모델 (Pydantic v2 최적화)"""

    role: str  # user, assistant, system
    content: str
    msg_type: str = "general"  # answer, log, greeting
    thought: str | None = None
    doc_ids: list[str] = []
    metrics: dict[str, Any] | None = None
    processed_content: str | None = None
    timestamp: float = Field(default_factory=time.time)

    @computed_field
    @property
    def formatted_time(self) -> str:
        """읽기 쉬운 형식의 시간 (계산된 필드)"""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    LangGraph 호환성을 위해 TypedDict 형식을 유지합니다.
    """

    input: str
    search_queries: list[str]
    documents: list[Document]
    relevant_docs: list[Document]  # [추가] 관련성 채점을 통과한 문서들
    context: str | None
    response: str | None
    thought: str | None  # [추가] 사고 과정
    performance: dict[str, Any] | None  # [추가] 성능 지표
    is_cached: bool  # [추가] 캐시 히트 여부


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


class AggregatedSearchResult(BaseModel):
    """통합 검색 결과 스키마 (Graph 내부용)"""

    doc_id: str
    content: str
    score: float
    node_id: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """검색 응답 스키마"""

    query: str
    results: list[SearchResult]
    count: int
