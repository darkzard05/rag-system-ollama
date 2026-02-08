"""
캐시 보안 관리 모듈.

다층 방어를 통해 pickle 역직렬화 공격으로부터 보호합니다:
1. 파일 무결성 검증 (SHA256)
2. 파일 권한 검사
3. HMAC 서명 (선택사항)
4. 신뢰 경로 화이트리스트
5. 자동 캐시 재생성
"""

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

    # 무결성 검증
    file_hash: str = Field(..., description="파일의 SHA256 해시 (16진수 문자열)")

    # 생성 시간
    created_at: str = Field(..., description="캐시 생성 시간 (ISO 8601 형식)")

    # 버전 관리
    cache_version: int = Field(
        default=2, description="캐시 포맷 버전 (1=레거시 pickle, 2=메타데이터 포함)"
    )

    # HMAC 서명 (선택)
    integrity_hmac: str | None = Field(
        default=None, description="HMAC-SHA256 서명 (base64)"
    )

    # 환경 정보
    python_version: str = Field(..., description="생성 환경의 Python 버전")

    rank_bm25_version: str | None = Field(
        default=None, description="rank-bm25 라이브러리 버전"
    )

    faiss_version: str | None = Field(default=None, description="faiss 라이브러리 버전")

    # 추가 메타데이터
    description: str | None = Field(
        default=None, description="캐시 설명 (예: PDF 파일명)"
    )

    @field_validator("file_hash")
    @classmethod
    def validate_hash_format(cls, v):
        """SHA256 해시 형식 검증"""
        if not isinstance(v, str) or len(v) != 64:
            raise ValueError("file_hash는 64자 16진수 문자열이어야 합니다")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("file_hash는 유효한 16진수 문자열이어야 합니다") from None
        return v

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, v):
        """ISO 8601 형식 검증"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise ValueError("created_at는 ISO 8601 형식이어야 합니다") from None
        return v


# ============================================================================
# 캐시 보안 관리자
# ============================================================================


class CacheSecurityManager:
    """
    캐시 파일의 보안을 관리하는 클래스.

    무결성 검증, 권한 검사, HMAC 서명을 담당합니다.
    """

    def __init__(
        self,
        security_level: str = "medium",
        hmac_secret: str | None = None,
        trusted_paths: list[str] | None = None,
        check_permissions: bool = True,
    ):
        """
        캐시 보안 관리자 초기화.

        Args:
            security_level: 보안 수준 ('low', 'medium', 'high')
            hmac_secret: HMAC 비밀 (고급 보안용, 32자 이상)
            trusted_paths: 신뢰할 수 있는 캐시 경로 목록
            check_permissions: 파일 권한 검사 여부

        Raises:
            ValueError: 유효하지 않은 보안 수준 또는 HMAC 비밀
        """
        if security_level not in ("low", "medium", "high"):
            raise ValueError(
                f"보안 수준은 'low', 'medium', 'high' 중 하나여야 합니다. "
                f"입력값: {security_level}"
            )

        if hmac_secret and len(hmac_secret) < 32:
            raise ValueError("HMAC 비밀은 최소 32자 이상이어야 합니다")

        self.security_level = security_level
        self.hmac_secret = hmac_secret
        self.trusted_paths = [Path(p).resolve() for p in (trusted_paths or [])]
        self.check_permissions = check_permissions

        logger.info(
            f"CacheSecurityManager 초기화: "
            f"level={security_level}, hmac={'활성' if hmac_secret else '비활성'}, "
            f"permission_check={check_permissions}"
        )

    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """
        파일의 해시를 계산합니다.

        Args:
            file_path: 파일 경로
            algorithm: 해시 알고리즘 ('sha256', 'sha512' 등)

        Returns:
            해시값 (16진수 문자열)

        Raises:
            FileNotFoundError: 파일이 없을 때
            IOError: 파일 읽기 실패
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        hasher = hashlib.new(algorithm)

        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError as e:
            logger.error(f"파일 해시 계산 실패: {file_path} - {e}")
            raise

    def compute_integrity_hmac(self, data: bytes, algorithm: str = "sha256") -> str:
        """
        데이터의 HMAC을 계산합니다.

        Args:
            data: 서명할 데이터
            algorithm: HMAC 알고리즘

        Returns:
            HMAC 값 (16진수 문자열)

        Raises:
            ValueError: HMAC 비밀이 설정되지 않았을 때
        """
        if not self.hmac_secret:
            raise ValueError("HMAC 서명을 위해 비밀이 설정되어야 합니다")

        # `hmac.new(..., digestmod=...)` expects either a callable/hash constructor
        # (e.g. hashlib.sha256) or an algorithm name string (e.g. "sha256").
        h = hmac.new(
            self.hmac_secret.encode(),
            data,
            digestmod=algorithm,
        )
        return h.hexdigest()

    @staticmethod
    def load_cache_metadata(metadata_path: str) -> CacheMetadata | None:
        """
        메타데이터 파일을 로드합니다.

        Args:
            metadata_path: 메타데이터 JSON 파일 경로

        Returns:
            CacheMetadata 객체 또는 None (파일 없을 때)
        """
        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                data = json.load(f)
            return CacheMetadata(**data)
        except Exception as e:
            logger.warning(f"메타데이터 로드 실패: {metadata_path} - {e}")
            return None

    def save_cache_metadata(self, metadata_path: str, metadata: CacheMetadata) -> None:
        """
        메타데이터 파일을 저장하고 권한을 강제합니다.

        Args:
            metadata_path: 메타데이터 JSON 파일 경로
            metadata: CacheMetadata 객체

        Raises:
            IOError: 파일 쓰기 실패
        """
        try:
            metadata_dir = os.path.dirname(metadata_path)
            if metadata_dir and not os.path.exists(metadata_dir):
                os.makedirs(metadata_dir, exist_ok=True)
                self.enforce_directory_permissions(metadata_dir)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)

            # 파일 권한 강제
            self.enforce_file_permissions(metadata_path)
            logger.debug(f"메타데이터 저장 및 보안 설정 완료: {metadata_path}")
        except OSError as e:
            logger.error(f"메타데이터 저장 실패: {metadata_path} - {e}")
            raise

    def verify_cache_integrity(
        self,
        file_path: str,
        metadata: CacheMetadata | None = None,
        metadata_path: str | None = None,
    ) -> bool:
        """
        캐시 파일의 무결성을 검증합니다.

        Args:
            file_path: 캐시 파일 경로
            metadata: CacheMetadata 객체 (없으면 metadata_path에서 로드)
            metadata_path: 메타데이터 파일 경로

        Returns:
            무결성 검증 성공 여부

        Raises:
            CacheIntegrityError: 무결성 검증 실패
            FileNotFoundError: 파일 없음
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"캐시 파일을 찾을 수 없습니다: {file_path}")

        # 메타데이터 로드
        if metadata is None:
            if metadata_path is None:
                metadata_path = file_path + ".meta"
            metadata = self.load_cache_metadata(metadata_path)

        if metadata is None:
            raise CacheIntegrityError(
                f"메타데이터가 없습니다. 캐시를 재생성해주세요: {file_path}"
            )

        # 파일 해시 검증
        try:
            current_hash = self.compute_file_hash(file_path)
            if current_hash != metadata.file_hash:
                raise CacheIntegrityError(
                    f"파일 해시 불일치 (파일 손상 또는 변조): {file_path}\n"
                    f"  예상: {metadata.file_hash}\n"
                    f"  실제: {current_hash}"
                )
        except CacheIntegrityError:
            raise
        except Exception as e:
            raise CacheIntegrityError(f"해시 계산 실패: {e}") from e

        # HMAC 검증
        if self.hmac_secret:
            if metadata.integrity_hmac:
                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    current_hmac = self.compute_integrity_hmac(file_data)

                    if not hmac.compare_digest(current_hmac, metadata.integrity_hmac):
                        raise CacheIntegrityError(
                            f"HMAC 검증 실패 (파일 변조 감지): {file_path}"
                        )
                    logger.debug(f"HMAC 검증 통과: {file_path}")
                except CacheIntegrityError:
                    raise
                except Exception as e:
                    logger.warning(f"HMAC 검증 중 오류: {e}")
            elif self.security_level in ("high", "medium"):
                # high 및 medium 레벨에서는 HMAC 필드가 있는 것이 권장됨 (high는 필수)
                msg = f"HMAC 서명이 누락되었습니다: {file_path}"
                if self.security_level == "high":
                    raise CacheIntegrityError(
                        f"보안 수준 'high'에서는 HMAC 서명이 필수입니다: {file_path}"
                    )
                else:
                    logger.warning(f"[Security] {msg}")
        elif self.security_level == "high":
            raise CacheIntegrityError(
                "보안 수준 'high'에서는 HMAC 비밀키가 필수입니다. CACHE_HMAC_SECRET을 설정하세요."
            )

        logger.debug(f"캐시 무결성 검증 통과: {file_path}")
        return True

    def check_file_permissions(self, file_path: str) -> bool:
        """
        파일 권한을 검사합니다.

        Args:
            file_path: 파일 경로

        Returns:
            권한 검사 통과 여부

        Raises:
            CachePermissionError: 권한 오류 (high 레벨)
            FileNotFoundError: 파일 없음
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        st = os.stat(file_path)
        file_mode = stat.S_IMODE(st.st_mode)

        # 다른 사용자가 읽을 수 있는 권한 확인
        others_readable = bool(file_mode & stat.S_IROTH)
        others_writable = bool(file_mode & stat.S_IWOTH)
        group_writable = bool(file_mode & stat.S_IWGRP)

        if others_readable or others_writable or group_writable:
            msg = (
                f"캐시 파일 권한이 너무 개방적입니다: {file_path}\n"
                f"  현재 권한: {oct(file_mode)}\n"
                f"  권장 권한: 0o600 또는 0o644"
            )

            if self.security_level == "high":
                raise CachePermissionError(msg)
            else:
                logger.warning(msg)
                return False

        logger.debug(f"파일 권한 검사 통과: {file_path} ({oct(file_mode)})")
        return True

    def check_directory_ownership(self, dir_path: str) -> bool:
        """
        디렉터리 소유권을 검사합니다 (Unix 환경).

        Args:
            dir_path: 디렉터리 경로

        Returns:
            소유권 검사 통과 여부

        Raises:
            CachePermissionError: 소유권 오류 (high 레벨)
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"디렉터리가 아닙니다: {dir_path}")

        # Windows에서는 소유권 검사 생략
        if os.name == "nt":
            return True

        st = os.stat(dir_path)
        # Windows에서는 getuid가 없으므로 getattr로 안전하게 접근하거나 환경 체크
        current_uid = getattr(os, "getuid", lambda: -1)()

        if current_uid != -1 and st.st_uid != current_uid:
            msg = (
                f"캐시 디렉터리 소유권이 일치하지 않습니다: {dir_path}\n"
                f"  소유자 UID: {st.st_uid}, 현재 UID: {current_uid}"
            )

            if self.security_level == "high":
                raise CachePermissionError(msg)
            else:
                logger.warning(msg)
                return False

        logger.debug(f"디렉터리 소유권 검사 통과: {dir_path}")
        return True

    def is_trusted_path(self, file_path: str) -> bool:
        """
        파일 경로가 신뢰 목록에 있는지 확인합니다.

        Args:
            file_path: 파일 경로

        Returns:
            신뢰할 수 있는 경로 여부
        """
        if not self.trusted_paths:
            # 신뢰 경로 화이트리스트가 없으면 모두 허용
            return True

        resolved_path = Path(file_path).resolve()

        for trusted in self.trusted_paths:
            try:
                # resolved_path가 trusted 디렉터리 아래인지 확인
                resolved_path.relative_to(trusted)
                logger.debug(f"신뢰 경로 확인 성공: {file_path}")
                return True
            except ValueError:
                continue

        return False

    def enforce_file_permissions(
        self, file_path: str, mode: int = CACHE_EXPECTED_FILE_MODE
    ) -> None:
        """
        파일의 권한을 보안 정책에 맞춰 강제 설정합니다.
        """
        try:
            os.chmod(file_path, mode)
            logger.debug(f"파일 권한 강제 설정 완료: {file_path} ({oct(mode)})")
        except Exception as e:
            logger.warning(f"파일 권한 설정 실패: {file_path} - {e}")

    def enforce_directory_permissions(
        self, dir_path: str, mode: int = CACHE_EXPECTED_DIR_MODE
    ) -> None:
        """
        디렉터리의 권한을 보안 정책에 맞춰 강제 설정합니다.
        """
        try:
            os.chmod(dir_path, mode)
            logger.debug(f"디렉터리 권한 강제 설정 완료: {dir_path} ({oct(mode)})")
        except Exception as e:
            logger.warning(f"디렉터리 권한 설정 실패: {dir_path} - {e}")

    def verify_cache_trust(self, file_path: str) -> bool:
        """
        캐시 파일을 신뢰할 수 있는지 종합적으로 검증합니다.

        Args:
            file_path: 캐시 파일 경로

        Returns:
            신뢰 가능 여부

        Raises:
            CacheTrustError: 신뢰할 수 없는 경로 (high 레벨)
        """
        if not self.is_trusted_path(file_path):
            msg = (
                f"캐시 파일이 신뢰할 수 없는 경로에 있습니다: {file_path}\n"
                f"  신뢰 경로: {self.trusted_paths}"
            )

            if self.security_level == "high":
                raise CacheTrustError(msg)
            else:
                logger.warning(msg)
                return False

        logger.debug(f"캐시 신뢰 검증 통과: {file_path}")
        return True

    def full_verification(
        self,
        file_path: str,
        metadata_path: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        캐시 파일에 대한 전체 보안 검증을 수행합니다.

        Args:
            file_path: 캐시 파일 경로
            metadata_path: 메타데이터 파일 경로

        Returns:
            (검증 성공 여부, 오류 메시지)
        """
        errors = []

        # 1. 신뢰 경로 검증
        try:
            self.verify_cache_trust(file_path)
        except CacheTrustError as e:
            errors.append(str(e))

        # 2. 파일 권한 검증
        try:
            self.check_file_permissions(file_path)
        except CachePermissionError as e:
            errors.append(str(e))
        except FileNotFoundError:
            pass  # 나중에 무결성 검증에서 처리

        # 3. 무결성 검증
        try:
            self.verify_cache_integrity(file_path, metadata_path=metadata_path)
        except (CacheIntegrityError, FileNotFoundError) as e:
            errors.append(str(e))

        if errors:
            error_msg = "\n".join(errors)
            return False, error_msg

        return True, None

    def create_metadata_for_file(
        self,
        file_path: str,
        description: str | None = None,
    ) -> CacheMetadata:
        """
        파일을 위한 새로운 메타데이터를 생성합니다. (HMAC 서명 포함 가능)

        Args:
            file_path: 캐시 파일 경로
            description: 캐시 설명

        Returns:
            CacheMetadata 객체
        """
        import sys

        try:
            import rank_bm25

            rank_bm25_version = rank_bm25.__version__
        except (ImportError, AttributeError):
            rank_bm25_version = None

        try:
            import faiss

            faiss_version = faiss.__version__
        except (ImportError, AttributeError):
            faiss_version = None

        file_hash = self.compute_file_hash(file_path)

        # HMAC 서명 생성 (비밀키가 있는 경우)
        integrity_hmac = None
        if self.hmac_secret:
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                integrity_hmac = self.compute_integrity_hmac(file_data)
            except Exception as e:
                logger.warning(f"HMAC 생성 실패: {e}")

        return CacheMetadata(
            file_hash=file_hash,
            integrity_hmac=integrity_hmac,
            created_at=datetime.now(timezone.utc).isoformat(),
            cache_version=2,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            rank_bm25_version=rank_bm25_version,
            faiss_version=faiss_version,
            description=description,
        )
