"""
config.yml 파일과 환경 변수에서 애플리케이션 설정을 로드합니다.
"""

import os
import yaml
from typing import Dict, List, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def _load_config() -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(
            "설정 파일(config.yml)을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요."
        )
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"설정 파일(config.yml)을 파싱하는 중 오류가 발생했습니다: {e}"
        )


# 설정 로드
_config = _load_config()

# --- 모델 및 설정 상수 ---
_models_config = _config.get("models", {})
OLLAMA_MODEL_NAME: str = _models_config.get("default_ollama", "qwen2:1.5b")
GEMINI_MODEL_NAME: str = _models_config.get("default_gemini", "gemini-1.5-flash")
OLLAMA_NUM_PREDICT: int = int(
    os.getenv("OLLAMA_NUM_PREDICT", _models_config.get("ollama_num_predict", -1))
)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
PREFERRED_GEMINI_MODELS: List[str] = _models_config.get("preferred_gemini", [])
AVAILABLE_EMBEDDING_MODELS: List[str] = _models_config.get("available_embeddings", [])
CACHE_DIR: str = _models_config.get("cache_dir", ".model_cache")

# --- RAG 파이프라인 설정 ---
_rag_config = _config.get("rag", {})
RETRIEVER_CONFIG: Dict = _rag_config.get("retriever", {})
TEXT_SPLITTER_CONFIG: Dict = _rag_config.get("text_splitter", {})
VECTOR_STORE_CACHE_DIR: str = _rag_config.get(
    "vector_store_cache_dir", ".model_cache/vector_store_cache"
)
_prompts_config = _rag_config.get("prompts") or {}
QA_SYSTEM_PROMPT: str = _prompts_config.get("qa_system_prompt", "")


# --- 채팅 UI 상수 ---
_ui_config = _config.get("ui", {})
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 650)
_ui_messages = _ui_config.get("messages", {})
MSG_PREPARING_ANSWER: str = _ui_messages.get("preparing_answer", "답변 생성 준비 중...")
MSG_NO_RELATED_INFO: str = _ui_messages.get(
    "no_related_info", "관련 정보를 찾을 수 없습니다."
)