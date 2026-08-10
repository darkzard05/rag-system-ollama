"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

from typing import Annotated, Any, Literal, TypedDict

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


def reset_or_append(existing: list[str], new: list[str]) -> list[str]:
    """턴 경계 리셋 기반 리스트 리듀서 (search_queries용).

    - `new`가 비어있으면 `[]`를 반환해 턴 시작 시 이전 턴의 재작성 쿼리 잔재를 제거한다.
    - `new`에 값이 있으면 `existing + new`를 반환해 동일 턴 내 재작성 쿼리를 누적한다.
    """
    if not new:
        return []
    return existing + new


def reset_or_add(existing: int, new: int) -> int:
    """턴 경계 리셋 기반 정수 리듀서 (retry_count용).

    - `new == 0`이면 `0`을 반환해 턴 시작 시 이전 턴의 재시도 예산을 리셋한다.
    - `new != 0`이면 `existing + new`를 반환해 동일 턴 내 재시도 횟수를 누적한다.
    """
    if new == 0:
        return 0
    return existing + new


class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 나타냅니다.
    LangGraph의 Reducer 기능을 활용하여 상태 업데이트를 관리합니다.
    """

    input: str
    intent: str | None
    route: Literal["generate", "transform"]  # R1a-06: 라우팅 전용 채널 (intent와 분리)
    search_queries: Annotated[list[str], reset_or_append]
    relevant_docs: list[Document]
    response: str | None
    thought: str | None
    performance: dict[str, Any] | None
    search_weights: dict[str, float] | None  # 👈 동적 가중치 추가
    is_cached: bool
    retry_count: Annotated[int, reset_or_add]


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


class LoginRequest(BaseModel):
    """로그인 요청 스키마"""

    username: str
    password: str


class LogoutRequest(BaseModel):
    """로그아웃 요청 스키마 (세션 ID는 선택)"""

    session_id: str | None = None


class TokenResponse(BaseModel):
    """인증 토큰 응답 스키마"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    session_id: str | None = None


class AggregatedSearchResult(BaseModel):
    """통합 검색 결과 스키마 (Graph 내부용)"""

    doc_id: str
    content: str
    score: float
    node_id: str
    metadata: dict[str, Any]
