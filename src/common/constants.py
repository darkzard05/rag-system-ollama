"""
애플리케이션 전체에서 사용되는 상수를 정의하는 모듈.

Enum 기반으로 관리하여 IDE 자동완성과 타입 검사를 지원합니다.
"""

from enum import IntEnum


class UIConstants(IntEnum):
    """UI 관련 상수 기초"""

    # 채팅 및 PDF 뷰어 기본 높이 (실제 화면 렌더링은 ui.py의 CSS calc(vh)가 덮어씌워 반응형으로 동작함)
    CONTAINER_HEIGHT = 600


class PerformanceConstants(IntEnum):
    """성능 관련 상수"""

    EMBEDDING_BATCH_SIZE_DEFAULT = 64
    EMBEDDING_BATCH_SIZE_GPU_HIGH = 128
    EMBEDDING_BATCH_SIZE_GPU_MID = 64
    EMBEDDING_BATCH_SIZE_GPU_LOW = 32
    EMBEDDING_BATCH_SIZE_CPU = 16
    MODEL_CACHE_TTL_SECONDS = 600
    MAX_MESSAGE_HISTORY = 1000


class ChunkingConstants(IntEnum):
    """문서 청킹 관련 상수"""

    MIN_CHUNK_SIZE = 200
    DEFAULT_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1000
    DEFAULT_OVERLAP_SIZE = 100
    MAX_HARD_SPLIT_LEN = 1500
    MIN_MERGE_LEN = 30
    SEMANTIC_WINDOW_SIZE = 3
    SIMILARITY_MERGE_THRESHOLD = 92


class RAGGraphConstants:
    """RAG 그래프 워크플로우 관련 상수"""

    CONTEXT_SAFE_BUDGET_RATIO = 0.6
    RERANK_MIN_DOCS_GUARANTEE = 3
    RERANK_DYNAMIC_THRESHOLD_RATIO = 0.6
    DEFAULT_BM25_WEIGHT = 0.4
    DEFAULT_FAISS_WEIGHT = 0.6


class RAGScoreConstants:
    """유사도 및 채점 관련 상수"""

    DEFAULT_MIN_SIMILARITY = 0.35
    EXTREME_SIMILARITY_THRESHOLD = 0.95


class TimeoutConstants(IntEnum):
    """타임아웃 관련 상수 (초 단위)"""

    RETRIEVER_TIMEOUT = 30
    LLM_TIMEOUT = 900
    QA_PIPELINE_TIMEOUT = 1200


class StringConstants:
    """문자열 상수"""

    PAGE_TITLE = "RAG Chatbot"
    LAYOUT = "wide"
    MAX_FILE_SIZE_MB = 50
    PDF_EXTENSION = ".pdf"


class FilePathConstants:
    """파일 경로 관련 상수"""

    LOG_DIR = "logs"
    LOG_FILE = "logs/app.log"
    TEMP_DIR = "data/temp"
