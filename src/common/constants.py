"""
애플리케이션 전체에서 사용되는 상수를 정의하는 모듈.

Enum 기반으로 관리하여 IDE 자동완성과 타입 검사를 지원합니다.
"""

from enum import IntEnum
from pathlib import Path

# 프로젝트 루트: src/common/constants.py → 루트 (config.py의 PROJECT_ROOT와 동일 공식)
# CWD와 무관하게 항상 루트 기준으로 해석되도록 절대경로를 사용합니다.
_PROJECT_ROOT = Path(__file__).parent.parent.parent


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
    """파일 경로 관련 상수 (모두 프로젝트 루트 기준 절대경로)"""

    # 로그 디렉터리 — CWD와 무관하게 항상 <루트>/logs/로 수렴
    LOG_DIR = _PROJECT_ROOT / "logs"
    LOG_FILE = _PROJECT_ROOT / "logs" / "app.log"

    # 캐시 디렉터리는 config.yml에서 로드하므로 여기서는 정의하지 않음

    # [추가] 전용 임시 디렉터리
    TEMP_DIR = _PROJECT_ROOT / "data" / "temp"
