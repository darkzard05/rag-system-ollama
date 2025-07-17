"""
애플리케이션의 모든 설정을 담는 파일입니다.
"""
import os
import yaml
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수를 로드합니다.

# --- YAML 설정 파일 로드 ---
def load_yaml_config(config_path: str) -> Dict:
    """YAML 설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: '{config_path}' not found. Using default values.")
        return {}
    except Exception as e:
        print(f"Error loading '{config_path}': {e}")
        return {}

# 현재 파일의 디렉토리 기준으로 config.yaml 경로 설정
_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
_yaml_config = load_yaml_config(_config_path)

# --- 모델 및 설정 상수 ---
OLLAMA_MODEL_NAME: str = "qwen3:4b"
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "-1")) # -1 for unlimited, or set a specific token limit

# Gemini API 키를 환경 변수에서 로드합니다.
# 예: export GEMINI_API_KEY="YOUR_API_KEY"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") 

# YAML 파일에서 Gemini 모델 목록 가져오기 (없으면 빈 리스트)
PREFERRED_GEMINI_MODELS: List[str] = _yaml_config.get('gemini_models', [])

AVAILABLE_EMBEDDING_MODELS: List[str] = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large-instruct",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2", # 진단을 위해 기본 모델로 변경
]
CACHE_DIR: str = ".model_cache"

# --- 리트리버 설정 상수 ---
RETRIEVER_CONFIG: Dict = {
    'search_type': "similarity",
    'search_kwargs': {
        'k': 5,
    },
    'weights': [0.4, 0.6]
}

# --- 텍스트 분할 설정 ---
TEXT_SPLITTER_CONFIG: Dict = {
    'chunk_size': 1500,
    'chunk_overlap': 150,
}

# --- 채팅 UI 상수 ---
THINK_START_TAG: str = "<think>"
THINK_END_TAG: str = "</think>"
MSG_PREPARING_ANSWER: str = "답변 생성 준비 중..."
MSG_THINKING: str = "🤔 생각을 정리하는 중입니다..."
MSG_NO_THOUGHT_PROCESS: str = "아직 생각 과정이 없습니다."
MSG_NO_RELATED_INFO: str = (
    "제공된 문서에서 질문에 대한 명확한 정보를 찾기 어렵습니다. 😥\n\n"
    "**다음을 시도해 보세요:**\n"
    "- 질문을 좀 더 명확하게 하거나 다른 키워드를 사용해 보세요.\n"
    "- 좀 더 일반적이거나 넓은 범위의 질문을 해보세요."
)
