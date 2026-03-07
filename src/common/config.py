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
# 하위 호환성을 위한 에일리어스
OLLAMA_MODEL_NAME = DEFAULT_OLLAMA_MODEL

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
OLLAMA_KEEP_ALIVE: str = _get_env(
    "OLLAMA_KEEP_ALIVE", _models_config.get("keep_alive", "30m"), str
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

# 최대 동시 추론 수 (VRAM 보호를 위한 전역 세마포어 값)
MAX_CONCURRENT_INFERENCE: int = _get_env(
    "MAX_CONCURRENT_INFERENCE", _models_config.get("max_concurrent_inference", 1), int
)

# 모델 캐시에서 유지할 모델의 최대 개수 (LRU 정책 적용)
MAX_CACHED_MODELS: int = _get_env(
    "MAX_CACHED_MODELS", _models_config.get("max_cached_models", 5), int
)

# --- RAG 파이프라인 설정 ---
_rag_config = _config.get("rag", {})
VECTOR_STORE_CONFIG: dict = _rag_config.get(
    "vector_store",
    {
        "engine": "faiss",
        "index_params": {
            "hnsw_m": 32,
            "quantization_threshold": 5000,
            "use_l2_norm": True,
            "distance_strategy": "MAX_INNER_PRODUCT",
        },
    },
)
RETRIEVER_CONFIG: dict = _rag_config.get("retriever", {})
# 앙상블 가중치 추출 (리트리버 설정 내에 존재)
ENSEMBLE_WEIGHTS: list[float] = RETRIEVER_CONFIG.get("ensemble_weights", [0.4, 0.6])

RERANKER_CONFIG: dict = _rag_config.get("reranker", {})
TEXT_SPLITTER_CONFIG: dict = _rag_config.get(
    "text_splitter", {"chunk_size": 500, "chunk_overlap": 100}
)
SEMANTIC_CHUNKER_CONFIG: dict = _rag_config.get("semantic_chunker", {"enabled": False})
PARSING_CONFIG: dict = _rag_config.get(
    "parsing",
    {
        "engine": "pymupdf4llm",
        "do_ocr": False,
        "do_table_structure": True,
        "table_strategy": "lines_strict",
        "timeout": 300.0,
        "extract_words": True,  # [수정] PDF 하이라이트 기능을 위해 좌표 추출 활성화
        "margins": (0, 72, 0, 72),  # 상하 1인치 마진 기본값
    },
)
VECTOR_STORE_CACHE_DIR: str = str(
    PROJECT_ROOT
    / _rag_config.get("vector_store_cache_dir", ".model_cache/vector_store_cache")
)
QUERY_EXPANSION_CONFIG: dict = _rag_config.get("query_expansion", {"enabled": True})
RAG_PARAMETERS: dict = _rag_config.get("parameters", {})
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

# --- 평가 설정 ---
_eval_config = _config.get("evaluation", {})
EVAL_JUDGE_MODEL: str = _get_env(
    "EVAL_JUDGE_MODEL", _eval_config.get("judge_model", DEFAULT_OLLAMA_MODEL)
)
EVAL_TIMEOUT: int = _get_env("EVAL_TIMEOUT", _eval_config.get("timeout", 1800), int)
EVAL_MAX_WORKERS: int = _get_env(
    "EVAL_MAX_WORKERS", _eval_config.get("max_workers", 1), int
)

_eval_prompts = _eval_config.get("prompts", {})
EVAL_PROMPT_ST_GEN: str = _eval_prompts.get("statement_generator", "")
EVAL_PROMPT_NLI: str = _eval_prompts.get("nli_statement", "")
EVAL_PROMPT_ANSWER_RELEVANCY: str = _eval_prompts.get("answer_relevancy", "")

_ui_errors = _ui_messages.get("errors", {})
MSG_ERROR_OLLAMA_NOT_RUNNING = _ui_errors.get(
    "ollama_not_running", "Ollama 서버 연결 실패"
)
