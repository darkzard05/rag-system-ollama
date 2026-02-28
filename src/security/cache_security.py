"""
캐시 보안 관리 모듈.

다층 방어를 통해 pickle 역직렬화 공격으로부터 보호합니다:
1. 파일 무결성 검증 (SHA256)
2. 파일 권한 검사
3. HMAC 서명 (선택사항)
4. 신뢰 경로 화이트리스트
5. 자동 캐시 재생성
"""

import contextlib
import hashlib
import hmac
import json
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from common.config import CACHE_EXPECTED_DIR_MODE, CACHE_EXPECTED_FILE_MODE

logger = logging.getLogger(__name__)


# ============================================================================
# 커스텀 예외
# ============================================================================


class CacheSecurityError(Exception):
    """캐시 보안 관련 기본 예외"""

    pass


class CacheIntegrityError(CacheSecurityError):
    """캐시 무결성 검증 실패"""

    pass


class CachePermissionError(CacheSecurityError):
    """캐시 파일 권한 오류"""

    pass


class CacheVersionError(CacheSecurityError):
    """캐시 버전 불일치"""

    pass


class CacheTrustError(CacheSecurityError):
    """신뢰할 수 없는 캐시 경로"""

    pass


# ============================================================================
# Pydantic 모델
# ============================================================================


class CacheMetadata(BaseModel):
    """
    캐시 메타데이터.
    Pickle 파일과 함께 저장되어 무결성/출처 검증에 사용됩니다.
    """

    file_hash: str = Field(..., description="파일의 SHA256 해시")
    created_at: str = Field(..., description="캐시 생성 시간")
    cache_version: int = Field(default=2, description="캐시 포맷 버전")
    integrity_hmac: str | None = Field(default=None, description="HMAC-SHA256 서명")
    python_version: str = Field(..., description="생성 환경의 Python 버전")
    rank_bm25_version: str | None = Field(default=None, description="rank-bm25 버전")
    faiss_version: str | None = Field(default=None, description="faiss 버전")
    description: str | None = Field(default=None, description="캐시 설명")

    @field_validator("file_hash")
    @classmethod
    def validate_hash_format(cls, v):
        if not isinstance(v, str) or len(v) != 64:
            raise ValueError("file_hash는 64자 16진수 문자열이어야 합니다")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError("created_at는 ISO 8601 형식이어야 합니다") from e
        return v


# ============================================================================
# 캐시 보안 관리자 (싱글톤)
# ============================================================================


class CacheSecurityManager:
    """
    캐시 파일의 보안을 관리하는 클래스 (싱글톤).
    """

    _instance: "CacheSecurityManager | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        security_level: str = "medium",
        hmac_secret: str | None = None,
        trusted_paths: list[str] | None = None,
        check_permissions: bool = True,
    ):
        if self._initialized:
            return

        self.security_level = security_level
        self.hmac_secret = hmac_secret
        self.trusted_paths = [Path(p).resolve() for p in (trusted_paths or [])]
        self.check_permissions = check_permissions
        self._initialized = True

        logger.debug(
            f"CacheSecurityManager 싱글톤 초기화 완료 (Level: {security_level})"
        )

    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일 없음: {file_path}")
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_integrity_hmac(self, data: bytes, algorithm: str = "sha256") -> str:
        if not self.hmac_secret:
            raise ValueError("HMAC 비밀키가 설정되지 않았습니다")
        h = hmac.new(self.hmac_secret.encode(), data, digestmod=algorithm)
        return h.hexdigest()

    @staticmethod
    def load_cache_metadata(metadata_path: str) -> CacheMetadata | None:
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, encoding="utf-8") as f:
                data = json.load(f)
            return CacheMetadata(**data)
        except Exception:
            return None

    def save_cache_metadata(self, metadata_path: str, metadata: CacheMetadata) -> None:
        try:
            metadata_dir = os.path.dirname(metadata_path)
            if metadata_dir and not os.path.exists(metadata_dir):
                os.makedirs(metadata_dir, exist_ok=True)
                self.enforce_directory_permissions(metadata_dir)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)
            self.enforce_file_permissions(metadata_path)
        except OSError as e:
            logger.error(f"메타데이터 저장 실패: {metadata_path} - {e}")
            raise

    def verify_cache_integrity(
        self,
        file_path: str,
        metadata: CacheMetadata | None = None,
        metadata_path: str | None = None,
    ) -> bool:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"캐시 파일 없음: {file_path}")
        if metadata is None:
            if metadata_path is None:
                metadata_path = file_path + ".meta"
            metadata = self.load_cache_metadata(metadata_path)
        if metadata is None:
            raise CacheIntegrityError(f"메타데이터 없음: {file_path}")

        current_hash = self.compute_file_hash(file_path)
        if current_hash != metadata.file_hash:
            raise CacheIntegrityError(f"해시 불일치: {file_path}")

        if self.hmac_secret and metadata.integrity_hmac:
            with open(file_path, "rb") as f:
                file_data = f.read()
            current_hmac = self.compute_integrity_hmac(file_data)
            if not hmac.compare_digest(current_hmac, metadata.integrity_hmac):
                raise CacheIntegrityError(f"HMAC 검증 실패: {file_path}")
        return True

    def check_file_permissions(self, file_path: str) -> bool:
        if os.name == "nt" or not self.check_permissions:
            return True
        st_info = os.stat(file_path)
        file_mode = stat.S_IMODE(st_info.st_mode)
        if bool(file_mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IWGRP)):
            if self.security_level == "high":
                raise CachePermissionError(f"권한 과다: {file_path}")
            return False
        return True

    def check_directory_ownership(self, dir_path: str) -> bool:
        if os.name == "nt":
            return True
        st_info = os.stat(dir_path)
        current_uid = getattr(os, "getuid", lambda: -1)()
        if current_uid != -1 and st_info.st_uid != current_uid:
            if self.security_level == "high":
                raise CachePermissionError(f"소유권 불일치: {dir_path}")
            return False
        return True

    def is_trusted_path(self, file_path: str) -> bool:
        if not self.trusted_paths:
            return True
        resolved_path = Path(file_path).resolve()
        for trusted in self.trusted_paths:
            try:
                resolved_path.relative_to(trusted)
                return True
            except ValueError:
                continue
        return False

    def enforce_file_permissions(
        self, file_path: str, mode: int = CACHE_EXPECTED_FILE_MODE
    ) -> None:
        if os.name == "nt":
            return
        with contextlib.suppress(Exception):
            os.chmod(file_path, mode)

    def enforce_directory_permissions(
        self, dir_path: str, mode: int = CACHE_EXPECTED_DIR_MODE
    ) -> None:
        if os.name == "nt":
            return
        with contextlib.suppress(Exception):
            os.chmod(dir_path, mode)

    def verify_cache_trust(self, file_path: str) -> bool:
        if not self.is_trusted_path(file_path):
            if self.security_level == "high":
                raise CacheTrustError(f"불신 경로: {file_path}")
            return False
        return True

    def full_verification(
        self, file_path: str, metadata_path: str | None = None
    ) -> tuple[bool, str | None]:
        try:
            self.verify_cache_trust(file_path)
            self.check_file_permissions(file_path)
            self.verify_cache_integrity(file_path, metadata_path=metadata_path)
            return True, None
        except Exception as e:
            return False, str(e)

    def create_metadata_for_file(
        self, file_path: str, description: str | None = None
    ) -> CacheMetadata:
        import sys

        file_hash = self.compute_file_hash(file_path)
        integrity_hmac = None
        if self.hmac_secret:
            with open(file_path, "rb") as f:
                file_data = f.read()
            integrity_hmac = self.compute_integrity_hmac(file_data)

        return CacheMetadata(
            file_hash=file_hash,
            integrity_hmac=integrity_hmac,
            created_at=datetime.now(timezone.utc).isoformat(),
            cache_version=2,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            description=description,
        )
