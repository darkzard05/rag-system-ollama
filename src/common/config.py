"""
config.yml 파일과 환경 변수에서 애플리케이션 설정을 로드합니다.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 설정 (현재 파일 기준 3단계 상위 디렉토리: src/common/config.py -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"


def _load_config() -> dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    try:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")

        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}  # 빈 파일일 경우 빈 딕셔너리 반환
    except Exception as e:
        # 설정 로드 실패는 치명적이므로 로그 남기고 재발생
        logger.critical(f"Failed to load configuration: {e}")
        raise RuntimeError(f"설정 파일 로드 실패: {e}") from e


def _get_env(key: str, default: Any, cast_type: Callable[[Any], Any] = str) -> Any:
    """환경 변수를 안전하게 가져오고 타입 변환합니다."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        cast_type_name = getattr(cast_type, "__name__", str(cast_type))
        logger.warning(
            f"환경 변수 '{key}'의 값 '{value}'을(를) {cast_type_name} 타입으로 변환할 수 없습니다. "
            f"기본값 '{default}'을(를) 사용합니다."
        )
        return default


# 설정 로드
_config = _load_config()

# --- 모델 및 설정 상수 ---
_models_config = _config.get("models", {})
DEFAULT_OLLAMA_MODEL: str = os.getenv(
    "DEFAULT_OLLAMA_MODEL",
    _models_config.get("default_ollama", "qwen3:4b-instruct-2507-q4_K_M"),
)

# Ollama 서버 주소 설정 (환경 변수 우선)
OLLAMA_BASE_URL: str = _get_env(
    "OLLAMA_BASE_URL", _models_config.get("base_url", "http://127.0.0.1:11434")
)

# 예측 파라미터 (환경 변수 우선, 실패 시 config.yml, 마지막으로 하드코딩된 기본값)
OLLAMA_NUM_PREDICT: int = _get_env(
    "OLLAMA_NUM_PREDICT", _models_config.get("ollama_num_predict", 4096), int
)
OLLAMA_TEMPERATURE: float = _get_env(
    "OLLAMA_TEMPERATURE", _models_config.get("temperature", 0.5), float
)
OLLAMA_NUM_CTX: int = _get_env(
    "OLLAMA_NUM_CTX", _models_config.get("num_ctx", 4096), int
)
OLLAMA_TOP_P: float = _get_env("OLLAMA_TOP_P", _models_config.get("top_p", 0.9), float)
OLLAMA_TIMEOUT: float = _get_env(
    "OLLAMA_TIMEOUT", _models_config.get("timeout", 900.0), float
)
# [최적화] 시스템 코어 수를 활용한 쓰레드 설정 (과부하 방지를 위해 보수적 할당)
OLLAMA_NUM_THREAD: int = _get_env(
    "OLLAMA_NUM_THREAD",
    max(1, (os.cpu_count() or 4) // 2),
    int,  # 코어의 절반만 사용
)

# --- 임베딩 설정 ---
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    _models_config.get("default_embedding", "nomic-embed-text"),
)
# 이제 임베딩 모델 목록은 기본적으로 default_embedding 하나만 포함합니다.
# (UI 등에서 Ollama 모델 목록을 병합하여 동적으로 확장 가능)
AVAILABLE_EMBEDDING_MODELS: list[str] = [DEFAULT_EMBEDDING_MODEL]
CACHE_DIR: str = str(PROJECT_ROOT / _models_config.get("cache_dir", ".model_cache"))
EMBEDDING_BATCH_SIZE: Union[int, str] = _models_config.get(
    "embedding_batch_size",
    16,  # auto 대신 명시적 값으로 메모리 제한
)
EMBEDDING_DEVICE: str = _models_config.get("embedding_device", "auto")

# --- RAG 파이프라인 설정 ---
_rag_config = _config.get("rag", {})
RETRIEVER_CONFIG: dict = _rag_config.get("retriever", {})
RERANKER_CONFIG: dict = _rag_config.get("reranker", {})
TEXT_SPLITTER_CONFIG: dict = _rag_config.get("text_splitter", {})
SEMANTIC_CHUNKER_CONFIG: dict = _rag_config.get("semantic_chunker", {})
VECTOR_STORE_CACHE_DIR: str = str(
    PROJECT_ROOT
    / _rag_config.get("vector_store_cache_dir", ".model_cache/vector_store_cache")
)
QUERY_EXPANSION_CONFIG: dict = _rag_config.get("query_expansion", {"enabled": True})
INTENT_ANALYSIS_ENABLED: bool = _rag_config.get("intent_analysis_enabled", False)
INTENT_PARAMETERS: dict = _rag_config.get("intent_parameters", {})
_prompts_config = _rag_config.get("prompts") or {}
ANALYSIS_PROTOCOL: str = _prompts_config.get("analysis_protocol", "")
RESEARCH_SYSTEM_PROMPT: str = _prompts_config.get("research_system_prompt", "")
FACTOID_SYSTEM_PROMPT: str = _prompts_config.get("factoid_system_prompt", "")
GREETING_SYSTEM_PROMPT: str = _prompts_config.get("greeting_system_prompt", "")
OUT_OF_CONTEXT_SYSTEM_PROMPT: str = _prompts_config.get(
    "out_of_context_system_prompt", ""
)
QA_SYSTEM_PROMPT: str = _prompts_config.get("qa_system_prompt", "")
QA_HUMAN_PROMPT: str = _prompts_config.get("qa_human_prompt", "")
QUERY_EXPANSION_PROMPT: str = _prompts_config.get("query_expansion_prompt", "")

# --- 캐시 보안 설정 ---
_cache_security_config = _config.get("cache_security", {})

# 보안 레벨 (environment variable 우선)
CACHE_SECURITY_LEVEL: str = _get_env(
    "CACHE_SECURITY_LEVEL", _cache_security_config.get("security_level", "medium"), str
)

# HMAC 비밀 (environment variable 우선)
CACHE_HMAC_SECRET: str | None = _get_env(
    "CACHE_HMAC_SECRET", _cache_security_config.get("hmac_secret"), str
)

# 신뢰 경로 (환경변수가 있으면 쉼표로 분리)
_trusted_paths_env = os.getenv("TRUSTED_CACHE_PATHS")
CACHE_TRUSTED_PATHS: list[str]
if _trusted_paths_env:
    CACHE_TRUSTED_PATHS = [p.strip() for p in _trusted_paths_env.split(",")]
else:
    CACHE_TRUSTED_PATHS = _cache_security_config.get("trusted_paths", [])

# 검증 실패 시 동작
CACHE_VALIDATION_ON_FAILURE: str = _get_env(
    "CACHE_VALIDATION_ON_FAILURE",
    _cache_security_config.get("on_validation_failure", "regenerate"),
    str,
)

# 파일 권한 검사
CACHE_CHECK_PERMISSIONS: bool = _get_env(
    "CACHE_CHECK_PERMISSIONS",
    _cache_security_config.get("check_permissions", True),
    lambda x: x.lower() == "true" if isinstance(x, str) else x,
)

# 예상 파일 권한
_expected_file_mode = _cache_security_config.get("expected_file_mode", 0o644)
CACHE_EXPECTED_FILE_MODE: int = (
    int(_expected_file_mode, 0)
    if isinstance(_expected_file_mode, str)
    else _expected_file_mode
)

_expected_dir_mode = _cache_security_config.get("expected_dir_mode", 0o755)
CACHE_EXPECTED_DIR_MODE: int = (
    int(_expected_dir_mode, 0)
    if isinstance(_expected_dir_mode, str)
    else _expected_dir_mode
)

# --- 전역 캐시 활성화 설정 ---
_cache_toggle_config = _config.get("global_cache", {})

ENABLE_VECTOR_CACHE: bool = _get_env(
    "ENABLE_VECTOR_CACHE",
    _cache_toggle_config.get("enable_vector_cache", True),
    lambda x: x.lower() == "true" if isinstance(x, str) else x,
)

ENABLE_RESPONSE_CACHE: bool = _get_env(
    "ENABLE_RESPONSE_CACHE",
    _cache_toggle_config.get("enable_response_cache", True),
    lambda x: x.lower() == "true" if isinstance(x, str) else x,
)


# --- 채팅 UI 상수 ---
_ui_config = _config.get("ui", {})
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 650)
_ui_messages = _ui_config.get("messages", {})

# UI 메시지 (get 메서드로 안전하게 가져오기)
MSG_PREPARING_ANSWER = _ui_messages.get("preparing_answer", "답변 생성 준비 중...")
MSG_NO_RELATED_INFO = _ui_messages.get(
    "no_related_info", "관련 정보를 찾을 수 없습니다."
)
MSG_SIDEBAR_TITLE = _ui_messages.get("sidebar_title", "⚙️ 설정")
MSG_PDF_UPLOADER_LABEL = _ui_messages.get("pdf_uploader_label", "PDF 파일 업로드")
MSG_MODEL_SELECTOR_LABEL = _ui_messages.get("model_selector_label", "LLM 모델 선택")
MSG_EMBEDDING_SELECTOR_LABEL = _ui_messages.get(
    "embedding_selector_label", "임베딩 모델 선택"
)
MSG_SYSTEM_STATUS_TITLE = _ui_messages.get("system_status_title", "📊 시스템 상태")
MSG_LOADING_MODELS = _ui_messages.get(
    "loading_models", "LLM 모델 목록을 불러오는 중..."
)
MSG_PDF_VIEWER_TITLE = _ui_messages.get("pdf_viewer_title", "📄 PDF 미리보기")
MSG_PDF_VIEWER_NO_FILE = _ui_messages.get(
    "pdf_viewer_no_file", "미리볼 PDF가 없습니다."
)
MSG_PDF_VIEWER_PREV_BUTTON = _ui_messages.get("pdf_viewer_prev_button", "← 이전")
MSG_PDF_VIEWER_NEXT_BUTTON = _ui_messages.get("pdf_viewer_next_button", "다음 →")
MSG_PDF_VIEWER_PAGE_SLIDER = _ui_messages.get("pdf_viewer_page_slider", "페이지 이동")
MSG_PDF_VIEWER_ERROR = _ui_messages.get("pdf_viewer_error", "PDF 오류: {e}")
MSG_CHAT_TITLE = _ui_messages.get("chat_title", "💬 채팅")
MSG_CHAT_INPUT_PLACEHOLDER = _ui_messages.get(
    "chat_input_placeholder", "PDF 내용에 대해 질문해보세요."
)
MSG_CHAT_NO_QA_SYSTEM = _ui_messages.get("chat_no_qa_system", "QA 시스템 미준비")
MSG_CHAT_GUIDE = _ui_messages.get("chat_guide", "사용 가이드")
MSG_STREAMING_ERROR = _ui_messages.get("streaming_error", "스트리밍 오류: {e}")
MSG_GENERIC_ERROR = _ui_messages.get("generic_error", "오류 발생: {error_msg}")
MSG_RETRY_BUTTON = _ui_messages.get("retry_button", "재시도")

_ui_errors = _ui_messages.get("errors", {})
MSG_ERROR_OLLAMA_NOT_RUNNING = _ui_errors.get(
    "ollama_not_running", "Ollama 서버 연결 실패"
)
