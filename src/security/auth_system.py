"""
Task 22-2: Authentication System
인증 및 토큰 관리 시스템
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class TokenType(Enum):
    """토큰 타입"""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    TEMPORARY = "temporary"


@dataclass
class Token:
    """토큰"""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_type: TokenType = TokenType.ACCESS
    user_id: str = ""
    token_string: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    last_used: float | None = None
    is_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """만료 여부"""
        if not self.expires_at:
            return False
        return time.time() > self.expires_at


@dataclass
class Session:
    """세션"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    token_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    ip_address: str | None = None
    user_agent: str | None = None
    is_active: bool = True

    def is_valid(self) -> bool:
        """유효 여부"""
        return self.is_active and time.time() < self.expires_at


class PasswordHasher:
    """비밀번호 해싱"""

    HASH_ALGORITHM = "sha256"
    ITERATIONS = 100000

    @staticmethod
    def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
        """비밀번호 해싱"""
        if not salt:
            salt = secrets.token_hex(32)

        # PBKDF2 시뮬레이션 (실제로는 bcrypt 등 사용)
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            PasswordHasher.ITERATIONS,
        )

        password_hash = base64.b64encode(hash_obj).decode("utf-8")

        return password_hash, salt

    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """비밀번호 검증"""
        computed_hash, _ = PasswordHasher.hash_password(password, salt)

        # 시간 공격 방지 (constant-time comparison)
        return hmac.compare_digest(computed_hash, stored_hash)


class SimpleJWT:
    """간단한 JWT 토큰 생성/검증"""

    def __init__(self, secret_key: str = "secret"):
        self.secret_key = secret_key

    def create_token(self, user_id: str, expires_in: int = 3600) -> str:
        """토큰 생성"""
        header = {"alg": "HS256", "typ": "JWT"}

        payload = {
            "user_id": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
        }

        # 헤더와 페이로드 인코딩
        header_encoded = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )

        payload_encoded = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        # 서명 생성
        message = f"{header_encoded}.{payload_encoded}"
        signature = hmac.new(
            self.secret_key.encode(), message.encode(), hashlib.sha256
        ).digest()

        signature_encoded = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{message}.{signature_encoded}"

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """토큰 검증"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_encoded, payload_encoded, signature_encoded = parts

            # 서명 검증
            message = f"{header_encoded}.{payload_encoded}"
            expected_signature = hmac.new(
                self.secret_key.encode(), message.encode(), hashlib.sha256
            ).digest()

            # 패딩 추가
            signature = base64.urlsafe_b64decode(signature_encoded + "===")

            if not hmac.compare_digest(signature, expected_signature):
                return None

            # 페이로드 디코딩
            payload_json = base64.urlsafe_b64decode(payload_encoded + "==").decode()

            payload = json.loads(payload_json)

            # 만료 확인
            if payload.get("exp", 0) < time.time():
                return None

            return payload

        except Exception:
            return None


class AuthenticationManager:
    """인증 관리자"""

    def __init__(self):
        self._users: dict[
            str, dict[str, Any]
        ] = {}  # user_id -> {password_hash, salt, ...}
        self._tokens: dict[str, Token] = {}
        self._sessions: dict[str, Session] = {}
        self._api_keys: dict[str, Token] = {}  # api_key_string -> Token object
        self._failed_logins: dict[str, list[float]] = {}  # user_id -> [timestamps]
        self._lock = RLock()
        self._max_failed_attempts = 5
        self._lockout_duration = 900  # 15분
        self._jwt = SimpleJWT(secret_key="your-secret-key")

    def register_user(self, user_id: str, username: str, password: str) -> bool:
        """사용자 등록"""
        with self._lock:
            if user_id in self._users:
                return False

            password_hash, salt = PasswordHasher.hash_password(password)

            self._users[user_id] = {
                "username": username,
                "password_hash": password_hash,
                "salt": salt,
                "created_at": time.time(),
                "last_login": None,
                "is_active": True,
            }

            return True

    def authenticate(
        self,
        user_id: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[str, str] | None:  # (access_token, session_id)
        """사용자 인증"""
        with self._lock:
            if user_id not in self._users:
                return None

            user_data = self._users[user_id]

            # 활성 여부 확인
            if not user_data.get("is_active"):
                return None

            # 잠금 확인
            if self._is_user_locked(user_id):
                return None

            # 비밀번호 검증
            if not PasswordHasher.verify_password(
                password, user_data["password_hash"], user_data["salt"]
            ):
                self._record_failed_login(user_id)
                return None

            # 성공
            self._clear_failed_logins(user_id)
            user_data["last_login"] = time.time()

            # 토큰 생성
            access_token_str = self._jwt.create_token(user_id, expires_in=3600)

            # 토큰 저장
            access_token = Token(
                token_type=TokenType.ACCESS,
                user_id=user_id,
                token_string=access_token_str,
                expires_at=time.time() + 3600,
            )
            self._tokens[access_token.token_id] = access_token

            # 세션 생성
            session = Session(
                user_id=user_id,
                token_id=access_token.token_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self._sessions[session.session_id] = session

            return access_token_str, session.session_id

    def _is_user_locked(self, user_id: str) -> bool:
        """사용자 잠금 여부"""
        if user_id not in self._failed_logins:
            return False

        recent_failures = [
            ts
            for ts in self._failed_logins[user_id]
            if time.time() - ts < self._lockout_duration
        ]

        return len(recent_failures) >= self._max_failed_attempts

    def _record_failed_login(self, user_id: str):
        """실패한 로그인 기록"""
        if user_id not in self._failed_logins:
            self._failed_logins[user_id] = []

        self._failed_logins[user_id].append(time.time())

    def _clear_failed_logins(self, user_id: str):
        """실패한 로그인 기록 초기화"""
        if user_id in self._failed_logins:
            del self._failed_logins[user_id]

    def verify_token(self, token_string: str) -> str | None:  # user_id
        """토큰 검증"""
        with self._lock:
            payload = self._jwt.verify_token(token_string)

            if not payload:
                return None

            user_id = payload.get("user_id")

            if not user_id or user_id not in self._users:
                return None

            return user_id

    def create_api_key(self, user_id: str, expires_in: int | None = None) -> str:
        """API 키 생성"""
        with self._lock:
            if user_id not in self._users:
                return ""

            api_key_str = f"sk_{secrets.token_urlsafe(32)}"
            expires_at = (time.time() + expires_in) if expires_in else None

            api_key_token = Token(
                token_type=TokenType.API_KEY,
                user_id=user_id,
                token_string=api_key_str,
                expires_at=expires_at,
            )
            self._api_keys[api_key_str] = api_key_token

            return api_key_str

    def register_fixed_api_key(
        self, user_id: str, api_key_str: str, expires_in: int | None = None
    ) -> bool:
        """[보안] 고정된 API 키 등록 (CI/Test 용)"""
        with self._lock:
            if user_id not in self._users:
                return False

            expires_at = (time.time() + expires_in) if expires_in else None
            api_key_token = Token(
                token_type=TokenType.API_KEY,
                user_id=user_id,
                token_string=api_key_str,
                expires_at=expires_at,
            )
            self._api_keys[api_key_str] = api_key_token
            return True

    def verify_api_key(self, api_key: str) -> str | None:  # user_id
        """API 키 검증 (만료 확인 포함)"""
        with self._lock:
            token_obj = self._api_keys.get(api_key)
            if not token_obj:
                return None

            if token_obj.is_expired() or not token_obj.is_valid:
                # 만료되거나 무효화된 키 자동 삭제
                del self._api_keys[api_key]
                return None

            token_obj.last_used = time.time()
            return token_obj.user_id

    def revoke_api_key(self, api_key: str) -> bool:
        """API 키 취소"""
        with self._lock:
            if api_key in self._api_keys:
                del self._api_keys[api_key]
                return True
            return False

    def get_session(self, session_id: str) -> Session | None:
        """세션 조회"""
        with self._lock:
            return self._sessions.get(session_id)

    def validate_session(self, session_id: str) -> bool:
        """세션 검증"""
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                return False

            if not session.is_valid():
                return False

            session.last_activity = time.time()
            return True

    def logout(self, session_id: str) -> bool:
        """로그아웃"""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.is_active = False
                return True
            return False

    def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> bool:
        """비밀번호 변경"""
        with self._lock:
            if user_id not in self._users:
                return False

            user_data = self._users[user_id]

            # 기존 비밀번호 검증
            if not PasswordHasher.verify_password(
                old_password, user_data["password_hash"], user_data["salt"]
            ):
                return False

            # 새 비밀번호 설정
            new_hash, new_salt = PasswordHasher.hash_password(new_password)
            user_data["password_hash"] = new_hash
            user_data["salt"] = new_salt

            return True

    def get_statistics(self) -> dict[str, Any]:
        """통계"""
        with self._lock:
            active_sessions = sum(1 for s in self._sessions.values() if s.is_valid())

            return {
                "total_users": len(self._users),
                "total_tokens": len(self._tokens),
                "active_sessions": active_sessions,
                "total_sessions": len(self._sessions),
                "api_keys": len(self._api_keys),
                "locked_users": sum(
                    1 for uid in self._failed_logins if self._is_user_locked(uid)
                ),
            }
