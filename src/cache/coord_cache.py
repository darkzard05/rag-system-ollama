"""
PDF 단어 좌표(word_coords)를 위한 사이드 캐시 매니저.
벡터 저장소의 메타데이터 비대화를 방지하기 위해 좌표 데이터를 별도로 저장하고 관리합니다.
"""

import logging
from functools import lru_cache
from pathlib import Path

import orjson

from common.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 캐시 디렉토리 설정
COORD_CACHE_DIR = PROJECT_ROOT / ".model_cache" / "coord_cache"


@lru_cache(maxsize=128)
def _load_from_file(file_hash: str, page_num: int) -> list[tuple] | None:
    """실제 파일 로딩 및 LRU 캐싱을 수행하는 독립 함수 (클래스 외부 정의로 메모리 누수 방지)."""
    cache_path = COORD_CACHE_DIR / f"{file_hash}_p{page_num}.json"
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            return orjson.loads(f.read())
    except Exception as e:
        logger.error(f"좌표 캐시 로드 실패 ({file_hash}, p{page_num}): {e}")
        return None


class CoordCacheManager:
    """단어 좌표 데이터를 파일 시스템에 캐싱하고 관리하는 클래스 (싱글톤)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        """매니저 초기화 및 디렉토리 생성."""
        if not COORD_CACHE_DIR.exists():
            COORD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"좌표 캐시 디렉토리 생성: {COORD_CACHE_DIR}")

    def _get_cache_path(self, file_hash: str, page_num: int) -> Path:
        """특정 페이지의 캐시 파일 경로를 반환합니다."""
        return COORD_CACHE_DIR / f"{file_hash}_p{page_num}.json"

    def save_coords(self, file_hash: str, page_num: int, coords: list[tuple]) -> bool:
        """좌표 데이터를 캐시에 저장합니다."""
        if not file_hash or not coords:
            return False

        cache_path = self._get_cache_path(file_hash, page_num)
        try:
            with open(cache_path, "wb") as f:
                f.write(orjson.dumps(coords))
            return True
        except Exception as e:
            logger.error(f"좌표 캐시 저장 실패 ({file_hash}, p{page_num}): {e}")
            return False

    def get_coords(self, file_hash: str, page_num: int) -> list[tuple] | None:
        """캐시에서 좌표 데이터를 로드합니다 (LRU 래퍼 호출)."""
        return _load_from_file(file_hash, page_num)

    def clear_cache(self, file_hash: str | None = None):
        """특정 파일 또는 전체 캐시를 삭제합니다."""
        try:
            if file_hash:
                for p in COORD_CACHE_DIR.glob(f"{file_hash}_p*.json"):
                    p.unlink()
            else:
                for p in COORD_CACHE_DIR.glob("*.json"):
                    p.unlink()

            # LRU 캐시 초기화
            _load_from_file.cache_clear()
        except Exception as e:
            logger.error(f"좌표 캐시 삭제 중 오류: {e}")


# 싱글톤 인스턴스 노출
coord_cache = CoordCacheManager()
