"""
애플리케이션 전체에서 사용되는 데이터 구조(스키마)를 정의합니다.
"""

import operator
import time
from datetime import datetime
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


class ChatMessage(BaseModel):
    """채팅 메시지 통합 모델 (Pydantic v2 최적화)"""

    role: str  # user, assistant, system
    content: str
    msg_type: str = "general"  # answer, log, greeting
    thought: str | None = None
    doc_ids: list[str] = []
    annotations: list[dict] = []  # [추가] 해당 메시지에 귀속된 PDF 하이라이트 좌표
    metrics: dict[str, Any] | None = None
    processed_content: str | None = None
    timestamp: float = Field(default_factory=time.time)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_time(self) -> str:
        """읽기 쉬운 형식의 시간 (계산된 필드)"""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


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


class ModelsConfig(BaseModel):
    default_ollama: str = "qwen3:4b-instruct-2507-q4_K_M"
    default_embedding: str = "nomic-embed-text-v2-moe"
    base_url: str = "http://127.0.0.1:11434"
    ollama_num_predict: int = -1
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    num_ctx: int = Field(default=4096, gt=0)
    model_ctx_overrides: dict[str, int] = Field(default_factory=dict)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    keep_alive: str = "30m"
    max_concurrent_inference: int = Field(default=1, gt=0)
    max_cached_models: int = Field(default=5, gt=0)
    embedding_batch_size: Any = "auto"
    embedding_device: str = "auto"
    cache_dir: str = ".model_cache"


class RetrieverConfig(BaseModel):
    search_type: str = "similarity"
    search_kwargs: dict[str, Any] = Field(default_factory=lambda: {"k": 5})
    ensemble_weights: list[float] = Field(default_factory=lambda: [0.4, 0.6])
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    dynamic_weighting: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = Field(default=15, gt=0)
    bypass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ParsingConfig(BaseModel):
    table_strategy: str = "lines_strict"
    graphics_limit: int = Field(default=5000, gt=0)
    fontsize_limit: int = Field(default=3, gt=0)
    ignore_code: bool = False
    hydration_mode: str = "precision_clip"
    margins: list[int] = Field(default_factory=lambda: [0, 72, 0, 72])
    write_images: bool = False


class SemanticChunkerConfig(BaseModel):
    enabled: bool = False
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_value: float = 95.0
    sentence_split_regex: str = r"[.!?]\s+"
    min_chunk_size: int = Field(default=100, gt=0)
    max_chunk_size: int = Field(default=800, gt=0)
    similarity_threshold: float = 0.5


class RAGConfig(BaseModel):
    vector_store_cache_dir: str = ".model_cache/vector_store_cache"
    vector_store: dict[str, Any] = Field(default_factory=dict)
    retriever: RetrieverConfig = Field(default_factory=lambda: RetrieverConfig())
    reranker: RerankerConfig = Field(default_factory=lambda: RerankerConfig())
    text_splitter: dict[str, int] = {"chunk_size": 500, "chunk_overlap": 100}
    parsing: ParsingConfig = Field(default_factory=lambda: ParsingConfig())
    semantic_chunker: SemanticChunkerConfig = Field(
        default_factory=lambda: SemanticChunkerConfig()
    )
    prompts: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=lambda: ModelsConfig())
    rag: RAGConfig = Field(default_factory=lambda: RAGConfig())
    evaluation: dict[str, Any] = Field(default_factory=dict)
    cache_security: dict[str, Any] = Field(default_factory=dict)
    global_cache: dict[str, Any] = Field(default_factory=dict)
    ui: dict[str, Any] = Field(default_factory=dict)
