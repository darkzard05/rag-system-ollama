"""
애플리케이션 전체에서 사용되는 상수를 정의하는 모듈.

Enum 기반으로 관리하여 IDE 자동완성과 타입 검사를 지원합니다.
"""

from enum import IntEnum


class PerformanceConstants(IntEnum):
    """성능 관련 상수"""

    # 임베딩 배치 처리
    EMBEDDING_BATCH_SIZE_DEFAULT = 64
    EMBEDDING_BATCH_SIZE_GPU_HIGH = 128
    EMBEDDING_BATCH_SIZE_GPU_MID = 64
    EMBEDDING_BATCH_SIZE_GPU_LOW = 32
    EMBEDDING_BATCH_SIZE_CPU = 16

    # 캐싱
    MODEL_CACHE_TTL_SECONDS = 600  # 10분


# 모듈 레벨 상수 (IntEnum 외부에서 직접 import 가능)
MAX_MESSAGE_HISTORY = 100


class ChunkingConstants(IntEnum):
    """문서 청킹 관련 상수"""

    # 청크 크기
    MIN_CHUNK_SIZE = 200
    DEFAULT_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1000

    # 청크 오버랩
    DEFAULT_OVERLAP_SIZE = 100

    # 의미론적 분할 (Semantic Chunking)
    MAX_HARD_SPLIT_LEN = 1500
    MIN_MERGE_LEN = 30
    SEMANTIC_WINDOW_SIZE = 3
    SIMILARITY_MERGE_THRESHOLD = 92  # 0.92 * 100 (IntEnum이므로 정수로 관리)


class StringConstants:
    """문자열 상수"""

    # 페이지 설정
    PAGE_TITLE = "RAG Chatbot"
    LAYOUT = "wide"

    # 파일 설정
    MAX_FILE_SIZE_MB = 50
    PDF_EXTENSION = ".pdf"


class FilePathConstants:
    """파일 경로 관련 상수"""

    # 로그 디렉터리
    LOG_DIR = "logs"
    LOG_FILE = "logs/app.log"

    # 캐시 디렉터리는 config.yml에서 로드하므로 여기서는 정의하지 않음

    # [추가] 전용 임시 디렉터리
    TEMP_DIR = "data/temp"
