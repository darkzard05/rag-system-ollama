"""
config.yml 파일과 환경 변수에서 애플리케이션 설정을 로드합니다.
"""

import logging
import os
import secrets
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


def _resolve_cache_hmac_secret(configured: str | None) -> str:
    """CACHE_HMAC_SECRET 값을 결정합니다.

    명시적으로 설정되지 않은 경우(None/빈 문자열) 암호학적으로 안전한 난수 키를
    생성하고 `.model_cache/.cache_hmac_key`에 영속화해 재시작 후에도 동일하게
    유지되도록 합니다. 영속 키가 없으면 매 재시작마다 기존 캐시 메타데이터 HMAC이
    무효화되므로 반드시 파일로 보존합니다.

    이 키 자동 생성은 pickle 직렬화 제거와 함께 RCE 경로를 차단하는
    defense-in-depth(다층 방어)입니다. HMAC 무결성 검증은 항상 활성화되어
    변조된 `.meta` 파일은 HMAC 불일치로 거부됩니다.
    ⚠️ 키 파일은 비공개로 유지되어야 합니다(.model_cache/ 는 .gitignore 대상).
    """
    if configured and configured.strip():
        return configured
    key_path = PROJECT_ROOT / ".model_cache" / ".cache_hmac_key"
    existing = None
    if key_path.exists():
        try:
            existing = key_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = None
    if existing:
        return existing
    generated = secrets.token_hex(32)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, generated.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        # 0o600 플래그 open이 실패하는 환경(일부 Windows)은 폴백으로 기록.
        # 기존 enforce_file_permissions와 동일하게 nt에서는 권한 오류를 무시.
        try:
            key_path.write_text(generated, encoding="utf-8")
        except OSError as e:
            logger.warning(f"HMAC 키 영속화 실패(메모리 전용으로 동작): {e}")
    return generated


_config = _load_config()

# --- 1. 모델 설정 (Models) ---
_models_config = _config.get("models", {})
DEFAULT_OLLAMA_MODEL: str = os.getenv(
    "DEFAULT_OLLAMA_MODEL",
    _models_config.get("default_ollama", "qwen3:4b-instruct-2507-q4_K_M"),
)
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
    "OLLAMA_NUM_CTX", _models_config.get("num_ctx", 8192), int
)
OLLAMA_TOP_P: float = _get_env("OLLAMA_TOP_P", _models_config.get("top_p", 0.8), float)
OLLAMA_THINKING: bool = _get_env(
    "OLLAMA_THINKING",
    _models_config.get("thinking", True),
    lambda x: str(x).lower() == "true",
)
OLLAMA_KEEP_ALIVE: str = _get_env(
    "OLLAMA_KEEP_ALIVE",
    _models_config.get("keep_alive", "30m"),
)
MODEL_CACHE_DIR: str = _get_env(
    "MODEL_CACHE_DIR",
    _models_config.get("cache_dir", ".model_cache"),
)
EMBEDDING_BATCH_SIZE: Union[int, str] = _get_env(
    "EMBEDDING_BATCH_SIZE",
    _models_config.get("embedding_batch_size", 16),
    lambda x: int(x) if str(x).isdigit() else x,
)
EMBEDDING_DEVICE: str = _get_env(
    "EMBEDDING_DEVICE", _models_config.get("embedding_device", "auto")
)
MAX_CACHED_MODELS: int = _get_env(
    "MAX_CACHED_MODELS", _models_config.get("max_cached_models", 5), int
)
# [WAVE4] 동시 추론 수는 int >= 1 여야 한다. 0 이하 또는 비정수면 로드 시점에
# 명확한 에러를 발생시킨다 (런타임 크래시나 1로의 묵시적 강제 변환 금지).
_MAX_CONCURRENT_INFERENCE_RAW: Any = _get_env(
    "MAX_CONCURRENT_INFERENCE",
    _models_config.get("max_concurrent_inference", 1),
    int,
)
if not isinstance(_MAX_CONCURRENT_INFERENCE_RAW, int) or isinstance(
    _MAX_CONCURRENT_INFERENCE_RAW, bool
):
    raise ValueError(
        "MAX_CONCURRENT_INFERENCE must be an integer, "
        f"got {_MAX_CONCURRENT_INFERENCE_RAW!r} "
        f"(type={type(_MAX_CONCURRENT_INFERENCE_RAW).__name__}). "
        "Set env MAX_CONCURRENT_INFERENCE or config key "
        "models.max_concurrent_inference to an integer >= 1."
    )
if _MAX_CONCURRENT_INFERENCE_RAW <= 0:
    raise ValueError(
        "MAX_CONCURRENT_INFERENCE must be an integer >= 1, "
        f"got {_MAX_CONCURRENT_INFERENCE_RAW}. "
        "OOM risk: values < 1 would disable the inference guard. "
        "Set env MAX_CONCURRENT_INFERENCE or config key "
        "models.max_concurrent_inference to an integer >= 1."
    )
MAX_CONCURRENT_INFERENCE: int = _MAX_CONCURRENT_INFERENCE_RAW
MAX_RESOURCE_POOL_SIZE: int = _get_env(
    "MAX_RESOURCE_POOL_SIZE", _models_config.get("max_resource_pool_size", 5), int
)
MAX_RESOURCE_POOL_SIZE_BYTES: int = _get_env(
    "MAX_RESOURCE_POOL_SIZE_BYTES",
    _models_config.get("max_resource_pool_size_bytes", 536870912),
    int,
)
OLLAMA_TIMEOUT: int = _get_env(
    "OLLAMA_TIMEOUT", _models_config.get("timeout", 120), int
)
ENABLE_OLLAMA_PRESSURE_FALLBACK: bool = _get_env(
    "ENABLE_OLLAMA_PRESSURE_FALLBACK",
    _models_config.get("enable_ollama_pressure_fallback", False),
    lambda x: str(x).lower() == "true",
)
HOST_PRESSURE_THRESHOLD: float = _get_env(
    "HOST_PRESSURE_THRESHOLD",
    _models_config.get("host_pressure_threshold", 85.0),
    float,
)

# --- 2. 임베딩 설정 (Embeddings) ---
_embedding_config = _config.get("embeddings", {})
DEFAULT_EMBEDDING_MODEL: str = _get_env(
    "DEFAULT_EMBEDDING_MODEL",
    _embedding_config.get("default_embedding", "nomic-embed-text-v2-moe"),
)
AVAILABLE_EMBEDDING_MODELS: list[str] = [DEFAULT_EMBEDDING_MODEL]

# --- 3. RAG 파이프라인 설정 (RAG) ---
_rag_config = _config.get("rag", {})
VECTOR_STORE_CONFIG: dict = _rag_config.get("vector_store", {})
RETRIEVER_CONFIG: dict = _rag_config.get("retriever", {})
DYNAMIC_WEIGHTING_CONFIG: dict = RETRIEVER_CONFIG.get(
    "dynamic_weighting", {"enabled": False}
)
ENSEMBLE_WEIGHTS: list[float] = RETRIEVER_CONFIG.get("ensemble_weights", [0.4, 0.6])
# R3a-04: RRF 점수 스케일(1/(k+rank)) 기준 동적 top-k 임계값 config (기본 gap 0.003)
DYNAMIC_TOP_K_CONFIG: dict = RETRIEVER_CONFIG.get(
    "dynamic_top_k",
    {"gap_threshold": 0.003, "min_candidates": 12, "max_candidates": 18},
)

_reranker_config = _rag_config.get("reranker", {})
RERANKER_MODEL_NAME: str = _reranker_config.get("model_name", "ms-marco-MultiBERT-L-12")
RERANKER_ENGINE: str = _reranker_config.get("engine", "auto")

TEXT_SPLITTER_CONFIG: dict = _rag_config.get(
    "text_splitter", {"chunk_size": 500, "chunk_overlap": 100}
)
SEMANTIC_CHUNKER_CONFIG: dict = _rag_config.get("semantic_chunker", {"enabled": False})

# 쿼리 캐시 설정 (선택적 opt-in)
_query_cache_config = _rag_config.get("query_cache", {})
QUERY_CACHE_ENABLED: bool = bool(_query_cache_config.get("enabled", False))
QUERY_CACHE_TTL: int = int(_query_cache_config.get("ttl_seconds", 3600))
QUERY_CACHE_MIN_CONF: float = float(_query_cache_config.get("min_confidence", 0.85))

# 채점(grading) 단계 설정
_grading_config = _rag_config.get("grading", {})
GRADING_ENABLED: bool = bool(_grading_config.get("enabled", True))

# --- 4. 파싱 및 하이드레이션 (Parsing) ---
PARSING_CONFIG: dict = _rag_config.get("parsing", {})
HYDRATION_MODE: str = PARSING_CONFIG.get("hydration_mode", "precision_clip")
VECTOR_STORE_CACHE_DIR: str = str(
    PROJECT_ROOT
    / _rag_config.get("vector_store_cache_dir", ".model_cache/vector_store_cache")
)

# --- 5. 프롬프트 설정 (Prompts) ---
# prompts 섹션은 config.yml의 rag.prompts 아래에 있음
_rag_config = _config.get("rag", {})
_prompts_config = _rag_config.get("prompts") or {}
ANALYSIS_PROTOCOL: str = _prompts_config.get("analysis_protocol", "")
GRADING_CONFIG: dict = _prompts_config.get("grading", {})
PROMPT_TEMPLATES_CONFIG: dict = _prompts_config.get("prompt_templates", {})

# --- 6. 보안 및 캐시 (Security & Global Cache) ---
_cache_security_config = _config.get("cache_security", {})
CACHE_SECURITY_LEVEL: str = _get_env(
    "CACHE_SECURITY_LEVEL", _cache_security_config.get("security_level", "medium"), str
)
CACHE_HMAC_SECRET: str | None = _resolve_cache_hmac_secret(
    _get_env("CACHE_HMAC_SECRET", _cache_security_config.get("hmac_secret"), str)
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

# --- 6.5 평가 설정 (Evaluation) ---
_evaluation_config = _config.get("evaluation", {})
EVAL_JUDGE_MODEL: str = _get_env(
    "EVAL_JUDGE_MODEL",
    _evaluation_config.get("judge_model", "qwen3:4b-instruct-2507-q4_K_M"),
)

# --- 7. UI 메시지 (UI) ---
_ui_config = _config.get("ui", {})

_ui_streaming = _ui_config.get("streaming", {})
UI_STREAMING_TIMEOUT: int = _ui_streaming.get("timeout_seconds", 30)

_ui_messages = _ui_config.get("messages", {})
MSG_CHAT_GUIDE: str = _ui_messages.get("chat_guide", "PDF를 업로드한 후 질문해 보세요")
MSG_PDF_VIEWER_NO_FILE: str = _ui_messages.get(
    "pdf_viewer_no_file", "No PDF to preview. Please upload a file from the sidebar."
)

MSG_ERROR_OLLAMA_NOT_RUNNING: str = _ui_config.get(
    "error_ollama_not_running",
    "Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.",
)

MSG_ERROR_EMBEDDING_FAILED: str = _ui_config.get(
    "error_embedding_failed",
    "임베딩 생성에 실패했습니다. 모델 설정이나 입력을 확인해 주세요.",
)

MSG_ERROR_VECTOR_STORE_FAILED: str = _ui_config.get(
    "error_vector_store_failed",
    "벡터 저장소 작업에 실패했습니다. 캐시 디렉터리 권한을 확인해 주세요.",
)

MSG_ERROR_GENERIC: str = _ui_config.get(
    "error_generic",
    "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
)

MSG_ERROR_PARSE_FAILED: str = _ui_config.get(
    "error_parse_failed",
    "문서 파싱에 실패했습니다. 파일 형식을 확인해 주세요.",
)

# --- 8. 에러 메시지 (Errors) ---
_error_config = _config.get("errors", {})
MSG_ERROR_OLLAMA_NOT_RUNNING = _error_config.get(
    "ollama_not_running", MSG_ERROR_OLLAMA_NOT_RUNNING
)
MSG_ERROR_EMBEDDING_FAILED = _error_config.get(
    "embedding_failed", MSG_ERROR_EMBEDDING_FAILED
)
MSG_ERROR_VECTOR_STORE_FAILED = _error_config.get(
    "vector_store_failed", MSG_ERROR_VECTOR_STORE_FAILED
)
MSG_ERROR_GENERIC = _error_config.get("generic", MSG_ERROR_GENERIC)
MSG_ERROR_PARSE_FAILED = _error_config.get("parse_failed", MSG_ERROR_PARSE_FAILED)

# --- 9. 검증 설정 (Verification) ---
_verification_config = _config.get("verification", {})
VERIFICATION_ENABLED: bool = _verification_config.get("enabled", False)
VERIFICATION_SAMPLE_RATE: float = _verification_config.get("sample_rate", 0.1)

# --- 10. 스트리밍 설정 (Streaming) ---
_streaming_config = _config.get("streaming", {})
