"""
설정 검증 모듈 (Pydantic 기반).

config.yml의 값들을 검증하여 타입 안정성과 제약 조건을 보장합니다.
설정 오류는 애플리케이션 시작 시 조기에 감지됩니다.

사용 예시:
    from common.config_validation import load_and_validate_config
    config = load_and_validate_config()
    # 또는
    from common.config_validation import ModelConfig
    model_cfg = ModelConfig(default_ollama="llama2", temperature=0.7)
"""

import logging
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """LLM 모델 설정."""

    default_ollama: str = Field(
        default="qwen3:4b-instruct-2507-q4_K_M", description="기본 Ollama 모델"
    )
    temperature: float = Field(
        default=0.3, ge=0.0, le=1.0, description="생성 온도 (0.0: 결정적, 1.0: 창의적)"
    )
    num_predict: int = Field(default=512, ge=1, le=8192, description="예측 토큰 수")
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling 확률"
    )
    num_ctx: int = Field(
        default=4096, ge=512, le=32000, description="컨텍스트 윈도우 크기"
    )
    timeout: float = Field(
        default=900.0, ge=10.0, le=3600.0, description="LLM 타임아웃 (초)"
    )

    @field_validator("temperature", "top_p")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """확률값 범위 검증."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"확률값은 0.0~1.0 사이여야 합니다: {v}")
        return v

    @field_validator("num_predict")
    @classmethod
    def validate_num_predict(cls, v: int) -> int:
        """예측 토큰 수 검증."""
        if v < 1:
            raise ValueError(f"예측 토큰 수는 1 이상이어야 합니다: {v}")
        if v > 8192:
            raise ValueError(f"예측 토큰 수는 8192 이하여야 합니다: {v}")
        return v


class EmbeddingConfig(BaseModel):
    """임베딩 모델 설정."""

    default_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="기본 임베딩 모델",
    )
    batch_size: int = Field(default=64, ge=1, le=512, description="배치 크기")
    cache_ttl: int = Field(default=600, ge=60, le=86400, description="캐시 TTL (초)")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """배치 크기 검증."""
        valid_sizes = [16, 32, 64, 128, 256, 512]
        if v not in valid_sizes:
            raise ValueError(f"배치 크기는 {valid_sizes} 중 하나여야 합니다: {v}")
        return v


class ChunkingConfig(BaseModel):
    """문서 청킹 설정."""

    chunk_size: int = Field(
        default=500, ge=100, le=2000, description="청크 크기 (문자)"
    )
    chunk_overlap: int = Field(
        default=100, ge=0, le=500, description="청크 오버랩 (문자)"
    )
    separator: str = Field(default="\n\n", description="청크 구분 문자")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "ChunkingConfig":
        """오버랩이 청크 크기보다 작은지 검증."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"오버랩({self.chunk_overlap})은 청크 크기({self.chunk_size})보다 작아야 합니다"
            )
        return self


class RetrievalConfig(BaseModel):
    """검색 설정."""

    top_k: int = Field(default=10, ge=1, le=100, description="상위 K개 문서 반환")
    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="유사도 임계값"
    )
    use_bm25: bool = Field(default=True, description="BM25 검색 사용")
    use_semantic: bool = Field(default=True, description="시멘틱 검색 사용")

    @model_validator(mode="after")
    def validate_retrieval_methods(self) -> "RetrievalConfig":
        """최소 하나의 검색 방식 선택 검증."""
        if not (self.use_bm25 or self.use_semantic):
            raise ValueError("BM25 또는 시멘틱 검색 중 최소 하나는 활성화되어야 합니다")
        return self


class CacheConfig(BaseModel):
    """캐시 설정."""

    cache_dir: str = Field(default=".model_cache", description="캐시 디렉터리 경로")
    cache_enabled: bool = Field(default=True, description="캐시 사용 여부")
    security_level: Literal["low", "medium", "high"] = Field(
        default="medium", description="캐시 보안 수준"
    )

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """캐시 디렉터리 검증."""
        if not v or len(v) == 0:
            raise ValueError("캐시 디렉터리는 비어있을 수 없습니다")
        if "/" in v and "\\" in v:
            raise ValueError("파이썬 경로 구분자를 섞어서 사용할 수 없습니다")
        return v


class UIConfig(BaseModel):
    """UI 설정."""

    title: str = Field(default="RAG Chatbot", description="페이지 제목")
    layout: Literal["wide", "centered"] = Field(
        default="wide", description="페이지 레이아웃"
    )
    max_upload_size_mb: int = Field(
        default=50, ge=1, le=500, description="최대 업로드 파일 크기 (MB)"
    )


class ApplicationConfig(BaseModel):
    """전체 애플리케이션 설정."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # 추가 필드 금지
    )

    def validate_all(self) -> tuple[bool, list[str]]:
        """모든 설정 검증.

        Returns:
            (is_valid, error_messages): 검증 결과 및 오류 메시지 목록
        """
        errors = []

        try:
            # 각 섹션 검증
            if self.model.temperature < 0 or self.model.temperature > 1:
                errors.append(
                    f"모델 온도가 범위를 벗어났습니다: {self.model.temperature}"
                )

            if self.embedding.batch_size < 1:
                errors.append(
                    f"임베딩 배치 크기가 너무 작습니다: {self.embedding.batch_size}"
                )

            if self.chunking.chunk_size < 100:
                errors.append(f"청크 크기가 너무 작습니다: {self.chunking.chunk_size}")

            if self.retrieval.top_k < 1:
                errors.append(f"top_k가 너무 작습니다: {self.retrieval.top_k}")

        except Exception as e:
            errors.append(f"검증 중 오류: {str(e)}")

        return len(errors) == 0, errors


def load_and_validate_config(config_dict: Optional[dict] = None) -> ApplicationConfig:
    """설정 로드 및 검증.

    Args:
        config_dict: 설정 딕셔너리 (None일 경우 기본값 사용)

    Returns:
        ApplicationConfig: 검증된 설정 객체

    Raises:
        ValueError: 설정 검증 실패 시
    """
    if config_dict is None:
        config_dict = {}

    try:
        config = ApplicationConfig(**config_dict)
        is_valid, errors = config.validate_all()

        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"설정 검증 실패:\n{error_msg}")

        logger.info("✅ 설정 검증 성공")
        return config

    except ValueError as e:
        logger.error(f"❌ 설정 검증 실패: {e}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)

    print("\n=== 설정 검증 테스트 ===\n")

    # 기본 설정으로 테스트
    try:
        config = load_and_validate_config()
        print("✅ 기본 설정 검증 성공")
        print("\n모델 설정:")
        print(f"  - LLM: {config.model.default_ollama}")
        print(f"  - 온도: {config.model.temperature}")
        print(f"  - 컨텍스트: {config.model.num_ctx}")
        print("\n임베딩 설정:")
        print(f"  - 모델: {config.embedding.default_model}")
        print(f"  - 배치 크기: {config.embedding.batch_size}")
        print("\n청킹 설정:")
        print(f"  - 청크 크기: {config.chunking.chunk_size}")
        print(f"  - 오버랩: {config.chunking.chunk_overlap}")
    except Exception as e:
        print(f"❌ 기본 설정 검증 실패: {e}")

    # 커스텀 설정으로 테스트
    print("\n\n--- 커스텀 설정 테스트 ---")
    custom_config = {
        "model": {
            "default_ollama": "llama2",
            "temperature": 0.5,
            "num_predict": 256,
        },
        "embedding": {
            "batch_size": 128,
        },
        "chunking": {
            "chunk_size": 400,
            "chunk_overlap": 50,
        },
    }

    try:
        config = load_and_validate_config(custom_config)
        print("✅ 커스텀 설정 검증 성공")
    except Exception as e:
        print(f"❌ 커스텀 설정 검증 실패: {e}")

    # 잘못된 설정으로 테스트
    print("\n\n--- 잘못된 설정 테스트 ---")
    invalid_config = {
        "model": {
            "temperature": 1.5,  # 범위 초과
        }
    }

    try:
        config = load_and_validate_config(invalid_config)
        print("✅ 검증 통과 (예기치 않음)")
    except Exception as e:
        print(f"✅ 예상된 오류 감지: {e}")
