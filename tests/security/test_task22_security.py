"""
Task 22: 보안 및 접근 제어 시스템 테스트
- RBAC (Role-Based Access Control)
- 인증 (Authentication) 및 토큰
- 암호화 (Encryption)
"""

import pytest
from src.security.auth_system import AuthenticationManager
from src.security.encryption_utils import (
    DataEncryptor,
    EncryptionManager,
    SecureDataStorage,
)
from src.security.rbac_system import RBAC, ActionType, ResourceType

# ==============================================
# Fixtures
# ==============================================


@pytest.fixture
def rbac():
    """RBAC Fixture"""
    return RBAC()


@pytest.fixture
def auth_manager():
    """인증 관리자 Fixture"""
    return AuthenticationManager()


@pytest.fixture
def data_encryptor():
    """데이터 암호화 Fixture"""
    return DataEncryptor()


# ==============================================
# 테스트 그룹 1: RBAC 기본 (5개 테스트)
# ==============================================


class TestRBACBasics:
    """RBAC 기본 테스트"""

    def test_01_create_permission(self, rbac):
        """권한 생성"""
        permission_id = rbac.create_permission(
            name="read_data", action=ActionType.READ, resource=ResourceType.DATA
        )

        assert permission_id is not None
        permission = rbac.get_permission(permission_id)
        assert permission.name == "read_data"

    def test_02_create_role(self, rbac):
        """역할 생성"""
        role_id = rbac.create_role(name="editor", description="Editor role")

        assert role_id is not None
        role = rbac.get_role(role_id)
        assert role.name == "editor"

    def test_03_add_permission_to_role(self, rbac):
        """역할에 권한 추가"""
        # 권한 생성
        perm_id = rbac.create_permission(
            name="write_data", action=ActionType.WRITE, resource=ResourceType.DATA
        )

        # 역할 생성
        role_id = rbac.create_role(name="editor")

        # 권한 추가
        result = rbac.add_permission_to_role(role_id, perm_id)

        assert result == True
        role = rbac.get_role(role_id)
        assert len(role.permissions) > 0

    def test_04_create_user(self, rbac):
        """사용자 생성"""
        user_id = rbac.create_user(username="john_doe", email="john@example.com")

        assert user_id is not None
        user = rbac.get_user(user_id)
        assert user.username == "john_doe"

    def test_05_assign_role_to_user(self, rbac):
        """사용자에게 역할 할당"""
        # 사용자 생성
        user_id = rbac.create_user(username="alice", email="alice@example.com")

        # 역할 조회 (기본 역할)
        role = rbac.get_role_by_name("admin")

        # 역할 할당
        result = rbac.assign_role_to_user(user_id, role.role_id)

        assert result == True
        user = rbac.get_user(user_id)
        assert len(user.roles) > 0


# ==============================================
# 테스트 그룹 2: 권한 검증 (5개 테스트)
# ==============================================


class TestAccessControl:
    """접근 제어 테스트"""

    def test_06_check_access_allowed(self, rbac):
        """접근 허용"""
        # 권한 생성
        perm_id = rbac.create_permission(
            name="read_config", action=ActionType.READ, resource=ResourceType.CONFIG
        )

        # 역할 생성 및 권한 추가
        role_id = rbac.create_role(name="reader")
        rbac.add_permission_to_role(role_id, perm_id)

        # 사용자 생성 및 역할 할당
        user_id = rbac.create_user(username="user1", email="user1@example.com")
        rbac.assign_role_to_user(user_id, role_id)

        # 접근 확인
        result = rbac.check_access(user_id, ActionType.READ, ResourceType.CONFIG)

        assert result == True

    def test_07_check_access_denied(self, rbac):
        """접근 거부"""
        # 권한 없음
        user_id = rbac.create_user(username="user2", email="user2@example.com")

        result = rbac.check_access(user_id, ActionType.DELETE, ResourceType.CONFIG)

        assert result == False

    def test_08_admin_access(self, rbac):
        """Admin 접근"""
        # 사용자 생성
        user_id = rbac.create_user(username="admin_user", email="admin@example.com")

        # Admin 역할 할당
        admin_role = rbac.get_role_by_name("admin")
        rbac.assign_role_to_user(user_id, admin_role.role_id)

        # 모든 접근 허용되어야 함
        result = rbac.check_access(user_id, ActionType.ADMIN, ResourceType.SYSTEM)

        assert result == True

    def test_09_lock_and_unlock_user(self, rbac):
        """사용자 잠금/해제"""
        user_id = rbac.create_user(username="user3", email="user3@example.com")

        # 사용자 잠금
        rbac.lock_user(user_id)
        user = rbac.get_user(user_id)
        assert user.is_locked == True

        # 접근 거부
        result = rbac.check_access(user_id, ActionType.READ, ResourceType.DATA)
        assert result == False

        # 사용자 잠금 해제
        rbac.unlock_user(user_id)
        user = rbac.get_user(user_id)
        assert user.is_locked == False

    def test_10_access_logs(self, rbac):
        """접근 로그"""
        user_id = rbac.create_user(username="user4", email="user4@example.com")

        # 여러 접근 시도
        rbac.check_access(user_id, ActionType.READ, ResourceType.DATA)
        rbac.check_access(user_id, ActionType.WRITE, ResourceType.DATA)
        rbac.check_access(user_id, ActionType.DELETE, ResourceType.DATA)

        # 로그 조회
        logs = rbac.get_access_logs(user_id=user_id)

        assert len(logs) >= 1


# ==============================================
# 테스트 그룹 3: 인증 (5개 테스트)
# ==============================================


class TestAuthentication:
    """인증 테스트"""

    def test_11_register_user(self, auth_manager):
        """사용자 등록"""
        result = auth_manager.register_user(
            user_id="user1", username="john", password="SecurePass123!"
        )

        assert result == True

    def test_12_authenticate_user(self, auth_manager):
        """사용자 인증"""
        # 사용자 등록
        auth_manager.register_user(
            user_id="user2", username="alice", password="Password123!"
        )

        # 인증
        result = auth_manager.authenticate(user_id="user2", password="Password123!")

        assert result is not None
        token, session_id = result
        assert token is not None
        assert session_id is not None

    def test_13_failed_authentication(self, auth_manager):
        """인증 실패"""
        # 사용자 등록
        auth_manager.register_user(
            user_id="user3", username="bob", password="CorrectPassword!"
        )

        # 잘못된 비밀번호로 인증
        result = auth_manager.authenticate(user_id="user3", password="WrongPassword!")

        assert result is None

    def test_14_verify_token(self, auth_manager):
        """토큰 검증"""
        # 사용자 등록 및 인증
        auth_manager.register_user(
            user_id="user4", username="charlie", password="Pass123!"
        )

        token_result = auth_manager.authenticate(user_id="user4", password="Pass123!")

        token, _ = token_result

        # 토큰 검증
        verified_user_id = auth_manager.verify_token(token)

        assert verified_user_id == "user4"

    def test_15_api_key_management(self, auth_manager):
        """API 키 관리"""
        # 사용자 등록
        auth_manager.register_user(
            user_id="user5", username="dave", password="Pass456!"
        )

        # API 키 생성
        api_key = auth_manager.create_api_key("user5")

        assert api_key is not None
        assert api_key.startswith("sk_")

        # API 키 검증
        verified_user = auth_manager.verify_api_key(api_key)
        assert verified_user == "user5"

        # API 키 취소
        result = auth_manager.revoke_api_key(api_key)
        assert result == True


# ==============================================
# 테스트 그룹 4: 암호화 (6개 테스트)
# ==============================================


class TestEncryption:
    """암호화 테스트"""

    def test_16_encrypt_decrypt_string(self, data_encryptor):
        """문자열 암호화/복호화"""
        plaintext = "This is a secret message"

        # 암호화
        encrypted = data_encryptor.encrypt_string(plaintext)

        assert encrypted != plaintext

        # 복호화
        decrypted = data_encryptor.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_17_encrypt_decrypt_dict(self, data_encryptor):
        """딕셔너리 암호화/복호화"""
        data = {
            "username": "john",
            "email": "john@example.com",
            "secret": "confidential",
        }

        # 암호화
        encrypted = data_encryptor.encrypt_dict(data)

        # 복호화
        decrypted = data_encryptor.decrypt_dict(encrypted)

        assert decrypted == data

    def test_18_hash_password(self, data_encryptor):
        """비밀번호 해싱"""
        password = "MySecurePassword123!"

        hash1 = data_encryptor.hash_password(password)
        hash2 = data_encryptor.hash_password(password)

        # 동일 비밀번호 = 동일 해시
        assert hash1 == hash2

    def test_19_encryption_manager(self):
        """암호화 관리자"""
        manager = EncryptionManager()

        # 민감한 데이터 암호화
        data = {
            "password": "secret123",
            "email": "user@example.com",
            "normal_field": "public_info",
        }

        encrypted_data = manager.encrypt_sensitive_data(data)

        # password와 email이 암호화됨
        assert isinstance(encrypted_data["password"], dict)
        assert encrypted_data["password"].get("encrypted") == True
        assert encrypted_data["normal_field"] == "public_info"

    def test_20_secure_data_storage(self):
        """보안 데이터 저장소"""
        storage = SecureDataStorage()

        # 민감한 데이터 저장
        secret = "This is a secret"
        storage.store("my_secret", secret, sensitive=True)

        # 데이터 조회
        retrieved = storage.retrieve("my_secret")

        assert retrieved == secret

        # 데이터 삭제
        result = storage.delete("my_secret")
        assert result == True

    def test_21_pii_masking(self):
        """개인식별정보 마스킹"""
        manager = EncryptionManager()

        # 이메일 마스킹
        masked = manager.encrypt_pii("john@example.com", method="mask")

        assert masked != "john@example.com"
        assert len(masked) == len("john@example.com")


# ==============================================
# 테스트 그룹 5: 통합 보안 (2개 테스트)
# ==============================================


class TestIntegratedSecurity:
    """통합 보안 테스트"""

    def test_22_end_to_end_security(self, auth_manager, rbac, data_encryptor):
        """엔드-투-엔드 보안"""
        # 1. 사용자 등록 (암호화된 비밀번호)
        password = "SecurePass123!"
        user_id = "secure_user"

        auth_manager.register_user(
            user_id=user_id, username="secure_user", password=password
        )

        # 2. 인증
        auth_result = auth_manager.authenticate(user_id, password)
        assert auth_result is not None

        token, session_id = auth_result

        # 3. 토큰 검증
        verified_user = auth_manager.verify_token(token)
        assert verified_user == user_id

        # 4. RBAC 확인
        rbac.create_user(username="secure_user", email="user@example.com")
        user_rbac = rbac.get_user_by_username("secure_user")

        if user_rbac:
            rbac.assign_role_to_user(
                user_rbac.user_id, rbac.get_role_by_name("admin").role_id
            )

            # 접근 확인
            can_access = rbac.check_access(
                user_rbac.user_id, ActionType.READ, ResourceType.CONFIG
            )
            assert can_access == True

    def test_23_security_statistics(self, auth_manager, rbac):
        """보안 통계"""
        # 사용자 생성
        for i in range(3):
            auth_manager.register_user(
                user_id=f"user_{i}", username=f"user_{i}", password="Pass123!"
            )

            rbac.create_user(username=f"rbac_user_{i}", email=f"user_{i}@example.com")

        # 통계
        auth_stats = auth_manager.get_statistics()
        rbac_stats = rbac.get_statistics()

        assert auth_stats["total_users"] >= 3
        assert rbac_stats["total_users"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
