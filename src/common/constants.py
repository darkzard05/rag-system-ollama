"""
애플리케이션 전체에서 사용되는 상수를 정의하는 모듈.

Enum 기반으로 관리하여 IDE 자동완성과 타입 검사를 지원합니다.
"""

from enum import IntEnum


class UIConstants(IntEnum):
    """UI 관련 상수"""

    # 채팅 및 PDF 뷰어 높이
    CONTAINER_HEIGHT = 650
    CHAT_SCROLL_HEIGHT = 650
    PDF_VIEWER_HEIGHT = 650


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

    # 메시지 히스토리
    MAX_MESSAGE_HISTORY = 1000


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


class RAGGraphConstants:
    """RAG 그래프 워크플로우 관련 상수"""

    # 컨텍스트 예산 (토큰 기준 비율)
    CONTEXT_SAFE_BUDGET_RATIO = 0.6

    # 리랭킹 및 필터링
    RERANK_MIN_DOCS_GUARANTEE = 3
    RERANK_DYNAMIC_THRESHOLD_RATIO = 0.6

    # 검색 통합 가중치
    DEFAULT_BM25_WEIGHT = 0.4
    DEFAULT_FAISS_WEIGHT = 0.6


class RAGScoreConstants:
    """유사도 및 채점 관련 상수"""

    # 기본 유사도 하한선
    DEFAULT_MIN_SIMILARITY = 0.35

    # 매우 높은 유사도 (중복 처리용)
    EXTREME_SIMILARITY_THRESHOLD = 0.95


class TimeoutConstants(IntEnum):
    """타임아웃 관련 상수 (초 단위)"""

    # 검색 작업
    RETRIEVER_TIMEOUT = 30

    # LLM 응답 생성
    LLM_TIMEOUT = 900  # 15분 (기존 5분에서 연장)

    # 전체 QA 파이프라인
    QA_PIPELINE_TIMEOUT = 1200  # 20분 (기존 10분에서 연장)


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
