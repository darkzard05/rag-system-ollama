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

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"


def _load_config() -> dict[str, Any]:
    try:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        raise RuntimeError(f"설정 파일 로드 실패: {e}") from e


def _get_env(key: str, default: Any, cast_type: Callable[[Any], Any] = str) -> Any:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default


_config = _load_config()

# --- 1. 모델 설정 (Models) ---
_models_config = _config.get("models", {})
DEFAULT_OLLAMA_MODEL: str = os.getenv(
    "DEFAULT_OLLAMA_MODEL",
    _models_config.get("default_ollama", "qwen3:4b-instruct-2507-q4_K_M"),
)
OLLAMA_MODEL_NAME = DEFAULT_OLLAMA_MODEL
OLLAMA_BASE_URL: str = _get_env(
    "OLLAMA_BASE_URL", _models_config.get("base_url", "http://127.0.0.1:11434")
)
OLLAMA_NUM_PREDICT: int = _get_env(
    "OLLAMA_NUM_PREDICT", _models_config.get("ollama_num_predict", 4096), int
)
OLLAMA_TEMPERATURE: float = _get_env(
    "OLLAMA_TEMPERATURE", _models_config.get("temperature", 0.1), float
)
OLLAMA_NUM_CTX: int = _get_env(
    "OLLAMA_NUM_CTX", _models_config.get("num_ctx", 4096), int
)
OLLAMA_TOP_P: float = _get_env("OLLAMA_TOP_P", _models_config.get("top_p", 0.8), float)
OLLAMA_KEEP_ALIVE: str = _get_env(
    "OLLAMA_KEEP_ALIVE", _models_config.get("keep_alive", "30m"), str
)

MAX_CONCURRENT_INFERENCE: int = _get_env(
    "MAX_CONCURRENT_INFERENCE", _models_config.get("max_concurrent_inference", 1), int
)
MAX_CACHED_MODELS: int = _get_env(
    "MAX_CACHED_MODELS", _models_config.get("max_cached_models", 5), int
)
MAX_RESOURCE_POOL_SIZE: int = _get_env(
    "MAX_RESOURCE_POOL_SIZE", _models_config.get("max_resource_pool_size", 10), int
)
MAX_RESOURCE_POOL_SIZE_BYTES: int = _get_env(
    "MAX_RESOURCE_POOL_SIZE_BYTES",
    _models_config.get("max_resource_pool_size_bytes", 2 * 1024 * 1024 * 1024),  # 2GB
    int,
)

# --- 2. 임베딩 설정 (Embeddings) ---
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    _models_config.get("default_embedding", "nomic-embed-text-v2-moe"),
)
AVAILABLE_EMBEDDING_MODELS: list[str] = [DEFAULT_EMBEDDING_MODEL]
CACHE_DIR: str = str(PROJECT_ROOT / _models_config.get("cache_dir", ".model_cache"))
EMBEDDING_BATCH_SIZE: Union[int, str] = _get_env(
    "EMBEDDING_BATCH_SIZE",
    _models_config.get("embedding_batch_size", 16),
    lambda x: int(x) if str(x).isdigit() else x,
)
EMBEDDING_DEVICE: str = _get_env(
    "EMBEDDING_DEVICE", _models_config.get("embedding_device", "auto")
)

# --- 3. RAG 파이프라인 설정 (RAG) ---
_rag_config = _config.get("rag", {})
VECTOR_STORE_CONFIG: dict = _rag_config.get("vector_store", {})
RETRIEVER_CONFIG: dict = _rag_config.get("retriever", {})
DYNAMIC_WEIGHTING_CONFIG: dict = RETRIEVER_CONFIG.get(
    "dynamic_weighting", {"enabled": False}
)
ENSEMBLE_WEIGHTS: list[float] = RETRIEVER_CONFIG.get("ensemble_weights", [0.4, 0.6])

_reranker_config = _rag_config.get("reranker", {})
RERANKER_MODEL_NAME: str = _reranker_config.get("model_name", "ms-marco-MultiBERT-L-12")
RERANKER_ENGINE: str = _reranker_config.get("engine", "auto")

TEXT_SPLITTER_CONFIG: dict = _rag_config.get(
    "text_splitter", {"chunk_size": 500, "chunk_overlap": 100}
)
SEMANTIC_CHUNKER_CONFIG: dict = _rag_config.get("semantic_chunker", {"enabled": False})

# --- 4. 파싱 및 하이드레이션 (Parsing) ---
PARSING_CONFIG: dict = _rag_config.get("parsing", {})
HYDRATION_MODE: str = PARSING_CONFIG.get("hydration_mode", "precision_clip")
VECTOR_STORE_CACHE_DIR: str = str(
    PROJECT_ROOT
    / _rag_config.get("vector_store_cache_dir", ".model_cache/vector_store_cache")
)

# --- 5. 프롬프트 설정 (Prompts) ---
_prompts_config = _rag_config.get("prompts") or {}
ANALYSIS_PROTOCOL: str = _prompts_config.get("analysis_protocol", "")
GRADING_CONFIG: dict = _prompts_config.get("grading", {})

# --- 6. 보안 및 캐시 (Security & Global Cache) ---
_cache_security_config = _config.get("cache_security", {})
CACHE_SECURITY_LEVEL: str = _get_env(
    "CACHE_SECURITY_LEVEL", _cache_security_config.get("security_level", "medium"), str
)
CACHE_HMAC_SECRET: str | None = _get_env(
    "CACHE_HMAC_SECRET", _cache_security_config.get("hmac_secret"), str
)
CACHE_TRUSTED_PATHS: list[str] = _cache_security_config.get("trusted_paths", [])
CACHE_CHECK_PERMISSIONS: bool = _get_env(
    "CACHE_CHECK_PERMISSIONS",
    _cache_security_config.get("check_permissions", True),
    lambda x: str(x).lower() == "true",
)

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

ENABLE_VECTOR_CACHE: bool = _get_env(
    "ENABLE_VECTOR_CACHE",
    _config.get("global_cache", {}).get("enable_vector_cache", True),
    lambda x: str(x).lower() == "true",
)

# --- 7. UI 메시지 (UI) ---
_ui_config = _config.get("ui", {})

_ui_streaming = _ui_config.get("streaming", {})
UI_STREAMING_TIMEOUT: int = _ui_streaming.get("timeout_seconds", 30)

UI_TIMELINE_POLL_SECONDS: float = float(_ui_config.get("timeline_poll_seconds", 1.0))

_ui_messages = _ui_config.get("messages", {})
MSG_PDF_VIEWER_NO_FILE = _ui_messages.get(
    "pdf_viewer_no_file", "미리볼 PDF가 없습니다."
)
MSG_CHAT_GUIDE: str = _ui_messages.get("chat_guide", "PDF를 업로드한 후 질문해 보세요")
_ui_errors = _ui_messages.get("errors", {})
MSG_ERROR_OLLAMA_NOT_RUNNING = _ui_errors.get(
    "ollama_not_running", "Ollama 서버 연결 실패"
)

# --- 8. 평가 (Evaluation) ---
_eval_config = _config.get("evaluation", {})
EVAL_JUDGE_MODEL: str = _get_env(
    "EVAL_JUDGE_MODEL", _eval_config.get("judge_model", DEFAULT_OLLAMA_MODEL)
)
