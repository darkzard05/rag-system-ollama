"""
Task 22: 보안 및 접근 제어 시스템 테스트
- 인증 (Authentication) 및 토큰

(삭제됨: RBAC / 암호화 — rbac_system, encryption_utils 모듈 제거로 동반 테스트 제거)
"""

import pytest
from src.security.auth_system import AuthenticationManager


@pytest.fixture
def auth_manager():
    """인증 관리자 Fixture"""
    return AuthenticationManager()


# ==============================================
# 테스트 그룹: 인증 (5개 테스트)
# ==============================================


class TestAuthentication:
    """인증 테스트"""

    def test_11_register_user(self, auth_manager):
        """사용자 등록"""
        result = auth_manager.register_user(
            user_id="user1", username="john", password="SecurePass123!"
        )

        assert result

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
        assert result


# ==============================================
# 테스트 그룹: 통합 보안 (2개 테스트)
# ==============================================


class TestIntegratedSecurity:
    """통합 보안 테스트"""

    def test_22_end_to_end_security(self, auth_manager):
        """엔드-투-엔드 보안 (인증 플로우)"""
        # 1. 사용자 등록 (해시된 비밀번호)
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

    def test_23_security_statistics(self, auth_manager):
        """보안 통계"""
        # 사용자 생성
        for i in range(3):
            auth_manager.register_user(
                user_id=f"user_{i}", username=f"user_{i}", password="Pass123!"
            )

        # 통계
        auth_stats = auth_manager.get_statistics()

        assert auth_stats["total_users"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
