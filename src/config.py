"""
애플리케이션의 모든 설정을 담는 파일입니다.
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수를 로드합니다.

# --- 모델 및 설정 상수 ---
OLLAMA_MODEL_NAME: str = "qwen3:4b"
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "-1")) # -1 for unlimited, or set a specific token limit

# Gemini API 키를 환경 변수에서 로드합니다.
# 예: export GEMINI_API_KEY="YOUR_API_KEY"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") 

AVAILABLE_EMBEDDING_MODELS: List[str] = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large-instruct",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2", # 진단을 위해 기본 모델로 변경
]
EMBEDDING_MODEL_NAME: str = AVAILABLE_EMBEDDING_MODELS[0]
CACHE_DIR: str = ".model_cache"

# --- 리트리버 설정 상수 ---
RETRIEVER_CONFIG: Dict = {
    'search_type': "similarity",
    'search_kwargs': {
        'k': 4,
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
MSG_WRITING_ANSWER: str = "답변을 작성하는 중..."
MSG_NO_THOUGHT_PROCESS: str = "아직 생각 과정이 없습니다."
MSG_NO_RELATED_INFO: str = "죄송합니다, 제공된 문서에서 관련 정보를 찾을 수 없었습니다."
