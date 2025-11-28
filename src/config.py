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
OLLAMA_MODEL_NAME: str = _models_config.get("default_ollama", "gemma3:8b")
# 예측 토큰 수
OLLAMA_NUM_PREDICT: int = int(
    os.getenv("OLLAMA_NUM_PREDICT", _models_config.get("ollama_num_predict", -1))
    )
# 온도 설정
OLLAMA_TEMPERATURE: float = float(
    os.getenv("OLLAMA_TEMPERATURE", _models_config.get("temperature", 0.5))
    )
# 컨텍스트 윈도우
OLLAMA_NUM_CTX: int = int(
    os.getenv("OLLAMA_NUM_CTX", _models_config.get("num_ctx", 2048)) 
)
# Top P
OLLAMA_TOP_P: float = float(
    os.getenv("OLLAMA_TOP_P", _models_config.get("top_p", 0.9))
)
AVAILABLE_EMBEDDING_MODELS: List[str] = _models_config.get("available_embeddings", [])
CACHE_DIR: str = _models_config.get("cache_dir", ".model_cache")
EMBEDDING_BATCH_SIZE: Any = _models_config.get("embedding_batch_size", "auto")

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
MSG_SIDEBAR_TITLE: str = _ui_messages.get("sidebar_title", "⚙️ 설정")
MSG_PDF_UPLOADER_LABEL: str = _ui_messages.get("pdf_uploader_label", "PDF 파일 업로드")
MSG_MODEL_SELECTOR_LABEL: str = _ui_messages.get("model_selector_label", "LLM 모델 선택")
MSG_EMBEDDING_SELECTOR_LABEL: str = _ui_messages.get(
    "embedding_selector_label", "임베딩 모델 선택"
)
MSG_SYSTEM_STATUS_TITLE: str = _ui_messages.get("system_status_title", "📊 시스템 상태")
MSG_LOADING_MODELS: str = _ui_messages.get(
    "loading_models", "LLM 모델 목록을 불러오는 중..."
)
MSG_PDF_VIEWER_TITLE: str = _ui_messages.get("pdf_viewer_title", "📄 PDF 미리보기")
MSG_PDF_VIEWER_NO_FILE: str = _ui_messages.get(
    "pdf_viewer_no_file", "미리볼 PDF가 없습니다. 사이드바에서 파일을 업로드해주세요."
)
MSG_PDF_VIEWER_PREV_BUTTON: str = _ui_messages.get("pdf_viewer_prev_button", "← 이전")
MSG_PDF_VIEWER_NEXT_BUTTON: str = _ui_messages.get("pdf_viewer_next_button", "다음 →")
MSG_PDF_VIEWER_PAGE_SLIDER: str = _ui_messages.get("pdf_viewer_page_slider", "페이지 이동")
MSG_PDF_VIEWER_ERROR: str = _ui_messages.get(
    "pdf_viewer_error", "PDF를 표시하는 중 오류가 발생했습니다: {e}"
)
MSG_CHAT_TITLE: str = _ui_messages.get("chat_title", "💬 채팅")
MSG_CHAT_INPUT_PLACEHOLDER: str = _ui_messages.get(
    "chat_input_placeholder", "PDF 내용에 대해 질문해보세요."
)
MSG_CHAT_NO_QA_SYSTEM: str = _ui_messages.get(
    "chat_no_qa_system", "QA 시스템이 준비되지 않았습니다. PDF를 먼저 처리해주세요."
)
MSG_CHAT_WELCOME: str = _ui_messages.get("chat_welcome", "환영합니다!")
MSG_CHAT_GUIDE: str = _ui_messages.get("chat_guide", "사용 가이드")
MSG_STREAMING_ERROR: str = _ui_messages.get(
    "streaming_error", "스트리밍 답변 생성 중 오류 발생: {e}"
)
MSG_GENERIC_ERROR: str = _ui_messages.get("generic_error", "오류가 발생했습니다: {error_msg}")
MSG_RETRY_BUTTON: str = _ui_messages.get("retry_button", "재시도")
_ui_errors = _ui_messages.get("errors", {})
MSG_ERROR_OLLAMA_NOT_RUNNING: str = _ui_errors.get(
    "ollama_not_running",
    "Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.",
)
