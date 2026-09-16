"""
DiskCache backend - HMAC-verified disk cache (L3).
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from threading import RLock

from services.optimization._backends_common import (
    CACHE_FORMAT,
    CACHE_FORMAT_KEY,
    CacheBackend,
    CacheEntry,
    T,
    _json_default,
)
from services.optimization._metrics import CacheStatistics

logger = logging.getLogger(__name__)


class DiskCache(CacheBackend[T]):
    """
    보안 강화된 디스크 기반 캐시 (L3)

    특징:
    - 영구 저장 지원
    - CacheSecurityManager를 통한 무결성 검증 (HMAC)
    - 역직렬화 전 보안 체크
    - TTL 기반 만료 처리
    """

    def __init__(self, cache_dir: str = "./.model_cache/response_cache"):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.stats = CacheStatistics()

        # 보안 관리자 초기화 (공유 인스턴스 사용으로 중복 로그 방지)
        from security.cache_security import get_security_manager

        self.security_manager = get_security_manager()

    def _get_cache_path(self, key: str) -> Path:
        """키에 대한 파일 경로 생성"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    async def get(self, key: str) -> T | None:
        """값 조회 (보안 검증 포함)"""
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> T | None:
        """동기 파일 조회 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            cache_file = self._get_cache_path(key)
            if not cache_file.exists():
                self.stats.total_misses += 1
                return None

            try:
                # 1. 보안 검증 (Full Verification)
                success, error = self.security_manager.full_verification(
                    str(cache_file)
                )
                if not success:
                    # [개선] 오류 성격에 따라 로그 레벨 조정
                    if "HMAC" in str(error) or "신뢰" in str(error):
                        logger.warning(
                            f"[DiskCache] 잠재적 보안 이슈로 캐시 무효화: {error}"
                        )
                    else:
                        logger.info(
                            f"[DiskCache] 유효하지 않은 캐시 항목 정리 (사유: {error})"
                        )

                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    return None

                # 2. 안전하게 로드 (unsafe 역직렬화 미사용 — JSON 전용)
                try:
                    with open(cache_file, encoding="utf-8") as f:
                        raw = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # 레거시 직렬화 파일 또는 손상된 파일 — 절대 역직렬화하지 않음
                    logger.info(
                        "[DiskCache] 레거시/손상된 캐시 파일 폐기 (JSON 파싱 실패)"
                    )
                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    return None

                if (
                    not isinstance(raw, dict)
                    or raw.get(CACHE_FORMAT_KEY) != CACHE_FORMAT
                ):
                    # 레거시/알 수 없는 포맷 — 폐기
                    logger.info(
                        "[DiskCache] 레거시/지원되지 않는 포맷의 캐시 파일 폐기"
                    )
                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    return None

                entry = CacheEntry.from_json_dict(raw)

                # 3. 만료 확인
                if entry.is_expired():
                    logger.debug(f"[DiskCache] 만료된 항목 제거: {key}")
                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    self.stats.total_expirations += 1
                    return None

                entry.touch()
                self.stats.total_hits += 1
                return entry.value

            except Exception as e:
                logger.error(f"[DiskCache] 로드 오류: {e}")
                # 손상된 캐시 파일 정리 (포맷 불일치/부분 쓰기 등)
                self._delete_file(cache_file)
                self.stats.total_misses += 1
                return None

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 저장 (보안 메타데이터 생성 및 권한 강제 포함)"""
        await asyncio.to_thread(self._set_sync, key, value, ttl_seconds)

    def _set_sync(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """동기 파일 저장 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            cache_file = self._get_cache_path(key)
            try:
                # 디렉토리 권한 보장
                if not self.cache_dir.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self.security_manager.enforce_directory_permissions(
                        str(self.cache_dir)
                    )

                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl_seconds if ttl_seconds > 0 else 86400.0,
                )

                # [최적화] 전체 디렉터리 스캔(glob) 대신 증분 카운터 유지 (쓰기당 O(N) 제거)
                if not cache_file.exists():
                    self.stats.cache_size += 1

                # 1. 파일 저장 — unsafe 역직렬화 대신 안전한 JSON 직렬화 (RCE 취약점 제거)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        entry.to_json_dict(),
                        f,
                        ensure_ascii=False,
                        indent=2,
                        default=_json_default,
                    )

                # 2. 파일 권한 강제 적용
                self.security_manager.enforce_file_permissions(str(cache_file))

                # 3. 보안 메타데이터 생성 및 저장
                metadata = self.security_manager.create_metadata_for_file(
                    str(cache_file), description=f"Cache entry: {key[:30]}"
                )
                self.security_manager.save_cache_metadata(
                    str(cache_file) + ".meta", metadata
                )

            except Exception as e:
                logger.error(f"[DiskCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        """동기 파일 삭제 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            self._delete_file(self._get_cache_path(key))

    async def clear(self) -> None:
        """전체 삭제"""
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        """동기 전체 삭제 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        import contextlib

        with self.lock:
            for f in self.cache_dir.glob("*.cache*"):
                with contextlib.suppress(Exception):
                    f.unlink()
            self.stats.cache_size = 0

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.update_hit_rate()
            return self.stats

    def _delete_file(self, path: Path) -> None:
        """파일 및 메타데이터 삭제 (증분 카운터 동기화 포함)"""
        try:
            removed = False
            if path.exists():
                path.unlink()
                removed = True
            meta = Path(str(path) + ".meta")
            if meta.exists():
                meta.unlink()
            if removed and self.stats.cache_size > 0:
                self.stats.cache_size -= 1
        except Exception:
            pass
