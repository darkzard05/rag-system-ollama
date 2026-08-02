"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, computed_field


class PerformanceStats(BaseModel):
    """통합 성능 메트릭 스키마 (Pydantic v2 최적화)"""

    ttft: float = 0.0
    thinking_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    thought_token_count: int = 0
    input_token_count: int = 0
    model_name: str = "unknown"
    doc_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tps(self) -> float:
        """Tokens Per Second (계산된 필드)"""
        return (
            self.token_count / self.generation_time if self.generation_time > 0 else 0.0
        )


class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    LangGraph의 Reducer 기능을 활용하여 상태 업데이트를 관리합니다.
    """

    input: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    intent: str | None
    search_queries: Annotated[list[str], operator.add]
    documents: list[Document]
    relevant_docs: list[Document]
    context: str | None
    response: str | None
    thought: str | None
    performance: dict[str, Any] | None
    search_weights: dict[str, float] | None  # 👈 동적 가중치 추가
    is_cached: bool
    retry_count: Annotated[int, operator.add]


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


class AggregatedSearchResult(BaseModel):
    """통합 검색 결과 스키마 (Graph 내부용)"""

    doc_id: str
    content: str
    score: float
    node_id: str
    metadata: dict[str, Any]
