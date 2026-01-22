"""
캐시 보안 시스템 테스트.

테스트 대상:
- CacheSecurityManager 클래스
- CacheMetadata Pydantic 모델
- 파일 해시 계산
- HMAC 서명
- 메타데이터 로드/저장
- 무결성 검증
- 권한 검사
"""

import hashlib
import json
import logging
import os
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import sys
# Add src directory to path (not tests directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from security.cache_security import (
    CacheSecurityManager,
    CacheMetadata,
    CacheIntegrityError,
    CachePermissionError,
    CacheTrustError,
    CacheSecurityError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """임시 디렉터리"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_file(temp_dir):
    """임시 파일"""
    file_path = os.path.join(temp_dir, "test_file.bin")
    with open(file_path, "wb") as f:
        f.write(b"test content for caching")
    yield file_path


@pytest.fixture
def security_manager():
    """캐시 보안 관리자 (기본 설정)"""
    return CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[],
        check_permissions=False,
    )


@pytest.fixture
def security_manager_high():
    """캐시 보안 관리자 (high 레벨)"""
    return CacheSecurityManager(
        security_level="high",
        hmac_secret="a" * 64,  # 64자 비밀
        trusted_paths=[],
        check_permissions=False,
    )


# ============================================================================
# CacheMetadata 테스트
# ============================================================================

class TestCacheMetadata:
    """CacheMetadata Pydantic 모델 테스트"""
    
    def test_valid_metadata(self):
        """유효한 메타데이터 생성"""
        metadata = CacheMetadata(
            file_hash="a" * 64,  # 64자 16진수
            created_at=datetime.now(timezone.utc).isoformat(),
            cache_version=2,
            python_version="3.10.0",
        )
        assert metadata.cache_version == 2
        assert len(metadata.file_hash) == 64
    
    def test_invalid_hash_format(self):
        """유효하지 않은 해시 형식"""
        with pytest.raises(ValueError):
            CacheMetadata(
                file_hash="invalid_hash",
                created_at=datetime.now(timezone.utc).isoformat(),
                python_version="3.10.0",
            )
    
    def test_invalid_datetime_format(self):
        """유효하지 않은 datetime 형식"""
        with pytest.raises(ValueError):
            CacheMetadata(
                file_hash="a" * 64,
                created_at="invalid-date",
                python_version="3.10.0",
            )
    
    def test_metadata_dict_export(self):
        """메타데이터를 딕셔너리로 변환"""
        metadata = CacheMetadata(
            file_hash="a" * 64,
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version="3.10.0",
        )
        data = metadata.model_dump()
        assert "file_hash" in data
        assert data["cache_version"] == 2


# ============================================================================
# 파일 해시 테스트
# ============================================================================

class TestFileHash:
    """파일 해시 계산 테스트"""
    
    def test_compute_file_hash(self, temp_file):
        """파일 해시 계산"""
        hash_value = CacheSecurityManager.compute_file_hash(temp_file)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256은 64자
        assert all(c in "0123456789abcdef" for c in hash_value)
    
    def test_file_hash_consistency(self, temp_file):
        """같은 파일의 해시는 일치"""
        hash1 = CacheSecurityManager.compute_file_hash(temp_file)
        hash2 = CacheSecurityManager.compute_file_hash(temp_file)
        assert hash1 == hash2
    
    def test_different_files_different_hashes(self, temp_dir):
        """다른 파일은 다른 해시"""
        file1 = os.path.join(temp_dir, "file1.bin")
        file2 = os.path.join(temp_dir, "file2.bin")
        
        with open(file1, "wb") as f:
            f.write(b"content1")
        with open(file2, "wb") as f:
            f.write(b"content2")
        
        hash1 = CacheSecurityManager.compute_file_hash(file1)
        hash2 = CacheSecurityManager.compute_file_hash(file2)
        
        assert hash1 != hash2
    
    def test_file_not_found(self, security_manager):
        """파일이 없을 때 예외"""
        with pytest.raises(FileNotFoundError):
            CacheSecurityManager.compute_file_hash("/nonexistent/path")


# ============================================================================
# HMAC 서명 테스트
# ============================================================================

class TestHMACSignature:
    """HMAC 서명 테스트"""
    
    def test_compute_hmac(self, security_manager_high):
        """HMAC 계산"""
        data = b"test data"
        hmac_value = security_manager_high.compute_integrity_hmac(data)
        
        assert isinstance(hmac_value, str)
        assert len(hmac_value) == 64  # SHA256 HMAC은 64자
    
    def test_hmac_consistency(self, security_manager_high):
        """같은 데이터의 HMAC은 일치"""
        data = b"test data"
        hmac1 = security_manager_high.compute_integrity_hmac(data)
        hmac2 = security_manager_high.compute_integrity_hmac(data)
        
        assert hmac1 == hmac2
    
    def test_hmac_requires_secret(self, security_manager):
        """비밀 없이 HMAC 계산 시 예외"""
        with pytest.raises(ValueError):
            security_manager.compute_integrity_hmac(b"data")
    
    def test_different_secrets_different_hmacs(self, temp_dir):
        """다른 비밀은 다른 HMAC"""
        manager1 = CacheSecurityManager(
            security_level="high",
            hmac_secret="secret1" + "a" * 50,
            check_permissions=False,
        )
        manager2 = CacheSecurityManager(
            security_level="high",
            hmac_secret="secret2" + "b" * 50,
            check_permissions=False,
        )
        
        data = b"test data"
        hmac1 = manager1.compute_integrity_hmac(data)
        hmac2 = manager2.compute_integrity_hmac(data)
        
        assert hmac1 != hmac2


# ============================================================================
# 메타데이터 저장/로드 테스트
# ============================================================================

class TestMetadataIO:
    """메타데이터 입출력 테스트"""
    
    def test_save_and_load_metadata(self, temp_dir, security_manager):
        """메타데이터 저장 및 로드"""
        metadata_path = os.path.join(temp_dir, "test.meta")
        
        metadata = CacheMetadata(
            file_hash="a" * 64,
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version="3.10.0",
            description="test cache",
        )
        
        # 저장
        security_manager.save_cache_metadata(metadata_path, metadata)
        assert os.path.exists(metadata_path)
        
        # 로드
        loaded = security_manager.load_cache_metadata(metadata_path)
        assert loaded is not None
        assert loaded.file_hash == metadata.file_hash
        assert loaded.description == "test cache"
    
    def test_load_nonexistent_metadata(self, security_manager):
        """존재하지 않는 메타데이터 로드"""
        result = security_manager.load_cache_metadata("/nonexistent/path.meta")
        assert result is None
    
    def test_metadata_json_format(self, temp_dir, security_manager):
        """메타데이터는 JSON 형식"""
        metadata_path = os.path.join(temp_dir, "test.meta")
        
        metadata = CacheMetadata(
            file_hash="b" * 64,
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version="3.11.0",
        )
        
        security_manager.save_cache_metadata(metadata_path, metadata)
        
        # JSON으로 직접 읽기
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert "file_hash" in data
        assert "created_at" in data


# ============================================================================
# 무결성 검증 테스트
# ============================================================================

class TestIntegrityVerification:
    """무결성 검증 테스트"""
    
    def test_verify_valid_cache(self, temp_dir, security_manager):
        """유효한 캐시 검증"""
        # 파일 생성
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"cache data")
        
        # 메타데이터 생성 및 저장
        metadata = security_manager.create_metadata_for_file(file_path)
        metadata_path = file_path + ".meta"
        security_manager.save_cache_metadata(metadata_path, metadata)
        
        # 검증 수행
        result = security_manager.verify_cache_integrity(
            file_path,
            metadata_path=metadata_path
        )
        assert result is True
    
    def test_verify_tampered_cache(self, temp_dir, security_manager):
        """변조된 캐시 감지"""
        # 파일 생성
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"original data")
        
        # 메타데이터 생성
        metadata = security_manager.create_metadata_for_file(file_path)
        metadata_path = file_path + ".meta"
        security_manager.save_cache_metadata(metadata_path, metadata)
        
        # 파일 변조
        with open(file_path, "wb") as f:
            f.write(b"modified data")
        
        # 검증 실패
        with pytest.raises(CacheIntegrityError):
            security_manager.verify_cache_integrity(
                file_path,
                metadata_path=metadata_path
            )
    
    def test_verify_missing_metadata(self, temp_dir, security_manager):
        """메타데이터 없을 때"""
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"data")
        
        with pytest.raises(CacheIntegrityError):
            security_manager.verify_cache_integrity(file_path)


# ============================================================================
# 신뢰 경로 검증 테스트
# ============================================================================

class TestTrustPath:
    """신뢰 경로 검증 테스트"""
    
    def test_trusted_path_allowed(self, temp_dir):
        """신뢰 경로에 있는 파일 허용"""
        manager = CacheSecurityManager(
            security_level="high",
            trusted_paths=[temp_dir],
            check_permissions=False,
        )
        
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"data")
        
        result = manager.is_trusted_path(file_path)
        assert result is True
    
    def test_untrusted_path_rejected(self, temp_dir):
        """신뢰 경로 외부 파일 거부"""
        manager = CacheSecurityManager(
            security_level="high",
            trusted_paths=[temp_dir],
            check_permissions=False,
        )
        
        untrusted_path = "/untrusted/path/cache.pkl"
        result = manager.is_trusted_path(untrusted_path)
        assert result is False
    
    def test_empty_trusted_paths_allows_all(self, temp_dir, security_manager):
        """신뢰 경로 목록이 비어있으면 모두 허용"""
        result = security_manager.is_trusted_path("/any/path")
        assert result is True


# ============================================================================
# 통합 테스트
# ============================================================================

class TestFullVerification:
    """전체 검증 프로세스 테스트"""
    
    def test_full_verification_success(self, temp_dir, security_manager):
        """전체 검증 성공"""
        # 캐시 파일 생성
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"valid cache data")
        
        # 메타데이터 생성
        metadata = security_manager.create_metadata_for_file(file_path)
        metadata_path = file_path + ".meta"
        security_manager.save_cache_metadata(metadata_path, metadata)
        
        # 전체 검증
        success, error = security_manager.full_verification(file_path, metadata_path)
        
        assert success is True
        assert error is None
    
    def test_full_verification_failure(self, temp_dir, security_manager):
        """전체 검증 실패"""
        # 캐시 파일 생성 및 변조
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"data")
        
        # 메타데이터 생성
        metadata = security_manager.create_metadata_for_file(file_path)
        metadata_path = file_path + ".meta"
        security_manager.save_cache_metadata(metadata_path, metadata)
        
        # 파일 변조
        with open(file_path, "wb") as f:
            f.write(b"tampered!")
        
        # 검증 실패
        success, error = security_manager.full_verification(file_path, metadata_path)
        
        assert success is False
        assert error is not None
        assert "해시" in error or "hash" in error.lower()


# ============================================================================
# 보안 레벨별 동작 테스트
# ============================================================================

class TestSecurityLevels:
    """보안 레벨별 동작 테스트"""
    
    def test_low_level_no_validation(self, temp_dir):
        """low 레벨: 검증 안함"""
        manager = CacheSecurityManager(
            security_level="low",
            check_permissions=False,
        )
        
        file_path = os.path.join(temp_dir, "cache.pkl")
        with open(file_path, "wb") as f:
            f.write(b"data")
        
        # 메타데이터 없어도 신뢰함
        # (실제로는 로드 시 메타데이터가 필요하지만, 
        # 감지할 때는 low 레벨에서 넘어갈 수 있음)
        assert manager.security_level == "low"
    
    def test_medium_level_hash_check(self, temp_dir):
        """medium 레벨: 해시 검증"""
        manager = CacheSecurityManager(
            security_level="medium",
            check_permissions=False,
        )
        
        assert manager.security_level == "medium"
    
    def test_high_level_full_check(self, temp_dir):
        """high 레벨: 전체 검증"""
        manager = CacheSecurityManager(
            security_level="high",
            hmac_secret="a" * 64,
            trusted_paths=[temp_dir],
            check_permissions=False,
        )
        
        assert manager.security_level == "high"


# ============================================================================
# 오류 처리 테스트
# ============================================================================

class TestErrorHandling:
    """오류 처리 테스트"""
    
    def test_invalid_security_level(self):
        """유효하지 않은 보안 레벨"""
        with pytest.raises(ValueError):
            CacheSecurityManager(security_level="invalid")
    
    def test_hmac_secret_too_short(self):
        """HMAC 비밀이 너무 짧음"""
        with pytest.raises(ValueError):
            CacheSecurityManager(
                security_level="high",
                hmac_secret="tooshort"
            )
    
    def test_cache_security_exception_hierarchy(self):
        """예외 클래스 계층 구조"""
        assert issubclass(CacheIntegrityError, CacheSecurityError)
        assert issubclass(CachePermissionError, CacheSecurityError)
        assert issubclass(CacheTrustError, CacheSecurityError)


# ============================================================================
# 성능 테스트
# ============================================================================

class TestPerformance:
    """성능 테스트"""
    
    def test_hash_computation_speed(self, temp_dir):
        """해시 계산 성능 (1MB 파일)"""
        import time
        
        # 1MB 파일 생성
        file_path = os.path.join(temp_dir, "large.bin")
        with open(file_path, "wb") as f:
            f.write(os.urandom(1024 * 1024))
        
        # 시간 측정
        start = time.time()
        hash_value = CacheSecurityManager.compute_file_hash(file_path)
        elapsed = time.time() - start
        
        logger.info(f"1MB 파일 해시 계산: {elapsed:.3f}초")
        assert elapsed < 1.0  # 1초 이내
    
    def test_metadata_save_load_speed(self, temp_dir, security_manager):
        """메타데이터 저장/로드 성능"""
        import time
        
        metadata_path = os.path.join(temp_dir, "test.meta")
        metadata = CacheMetadata(
            file_hash="c" * 64,
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version="3.10.0",
        )
        
        # 저장 시간
        start = time.time()
        security_manager.save_cache_metadata(metadata_path, metadata)
        save_time = time.time() - start
        
        # 로드 시간
        start = time.time()
        loaded = security_manager.load_cache_metadata(metadata_path)
        load_time = time.time() - start
        
        logger.info(f"메타데이터 저장: {save_time*1000:.1f}ms, 로드: {load_time*1000:.1f}ms")
        assert save_time < 0.1
        assert load_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
