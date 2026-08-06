"""
Task 22-2: Authentication System
인증 및 토큰 관리 시스템
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


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


def _token_to_dict(t: Token) -> dict[str, Any]:
    """Token 을 JSON 직렬화 가능한 dict 로 변환합니다."""
    d = t.__dict__.copy()
    d["token_type"] = t.token_type.value
    return d


def _token_from_dict(d: dict[str, Any]) -> Token:
    """직렬화된 dict 로부터 Token 을 복원합니다."""
    d = dict(d)
    d["token_type"] = TokenType(d["token_type"])
    return Token(**d)


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

    def __init__(self, secret_key: str | None = None, secret_file: str | None = None):
        self.secret_key = (
            secret_key
            or os.getenv("JWT_SECRET_KEY")
            or self._load_or_create_secret(secret_file)
        )

    @staticmethod
    def _load_or_create_secret(secret_file: str | None) -> str:
        """시크릿 파일을 로드하거나 생성합니다 (없으면 무작위 키 반환)."""
        if not secret_file:
            return secrets.token_urlsafe(32)
        path = Path(secret_file)
        try:
            if path.exists() and path.stat().st_size > 0:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    return content

            path.parent.mkdir(parents=True, exist_ok=True)
            secret = secrets.token_urlsafe(32)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(secret)
            os.replace(tmp_path, path)
            with suppress(OSError):
                os.chmod(path, 0o600)
            return secret
        except OSError:
            return secrets.token_urlsafe(32)

    def create_token(self, user_id: str, expires_in: int = 3600) -> str:
        """토큰 생성"""
        header = {"alg": "HS256", "typ": "JWT"}

        payload = {
            "user_id": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "jti": str(uuid.uuid4()),
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

    def __init__(self, state_file: str | None = None, secret_file: str | None = None):
        self._state_file = (
            state_file
            or os.getenv("AUTH_STATE_FILE")
            or str(Path(".model_cache") / "auth_state.json")
        )
        self._secret_file = (
            secret_file
            or os.getenv("AUTH_SECRET_FILE")
            or str(Path(".model_cache") / ".jwt_secret")
        )
        self._users: dict[
            str, dict[str, Any]
        ] = {}  # user_id -> {password_hash, salt, ...}
        self._tokens: dict[str, Token] = {}
        self._tokens_by_string: dict[str, Token] = {}  # token_string -> Token
        self._sessions: dict[str, Session] = {}
        self._api_keys: dict[str, Token] = {}  # api_key_string -> Token object
        self._failed_logins: dict[str, list[float]] = {}  # user_id -> [timestamps]
        self._deny_list: set[str] = set()
        self._lock = RLock()
        self._max_failed_attempts = 5
        self._lockout_duration = 900  # 15분
        self._load_state()
        self._jwt = SimpleJWT(secret_file=self._secret_file)

    def _load_state(self) -> None:
        """디스크에서 인증 상태를 복원합니다 (실패 시 무시)."""
        path = Path(self._state_file)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        users = data.get("users") or {}
        if isinstance(users, dict):
            self._users = users

        sessions = data.get("sessions") or {}
        restored_sessions: dict[str, Session] = {}
        for sid, s in sessions.items():
            try:
                session = Session(**s)
            except TypeError:
                continue
            if session.is_valid():
                restored_sessions[sid] = session
        self._sessions = restored_sessions

        api_keys = data.get("api_keys") or {}
        restored_keys: dict[str, Token] = {}
        for key, t in api_keys.items():
            try:
                restored_keys[key] = _token_from_dict(t)
            except (TypeError, ValueError):
                continue
        self._api_keys = restored_keys

        self._deny_list = set(data.get("deny_list", []))

    def _save_state(self) -> None:
        """인증 상태를 디스크에 원자적으로 저장합니다 (실패 시 경고만)."""
        try:
            payload = {
                "users": self._users,
                "sessions": {sid: s.__dict__ for sid, s in self._sessions.items()},
                "api_keys": {
                    key: _token_to_dict(t) for key, t in self._api_keys.items()
                },
                "deny_list": sorted(self._deny_list),
            }
            path = Path(self._state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError as e:
            logger.warning("auth state 저장 실패 (%s): %s", self._state_file, e)

    def register_user(
        self,
        user_id: str,
        username: str,
        password: str,
        role: str = "user",
    ) -> bool:
        """사용자 등록"""
        with self._lock:
            if user_id in self._users:
                return False

            password_hash, salt = PasswordHasher.hash_password(password)

            self._users[user_id] = {
                "username": username,
                "password_hash": password_hash,
                "salt": salt,
                "role": role,
                "created_at": time.time(),
                "last_login": None,
                "is_active": True,
            }

            self._save_state()
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
            self._tokens_by_string[access_token_str] = access_token

            # 세션 생성
            session = Session(
                user_id=user_id,
                token_id=access_token.token_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self._sessions[session.session_id] = session

            self._save_state()
            return access_token_str, session.session_id

    def is_admin(self, user_id: str) -> bool:
        """관리자 역할 여부"""
        with self._lock:
            user_data = self._users.get(user_id)
            return bool(user_data and user_data.get("role") == "admin")

    def authenticate_by_username(
        self,
        username: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[str, str] | None:
        """사용자 이름 기반 인증 (로그인 엔드포인트용)"""
        with self._lock:
            user_id = next(
                (
                    uid
                    for uid, u in self._users.items()
                    if u.get("username") == username
                ),
                None,
            )
        if user_id is None:
            return None
        return self.authenticate(user_id, password, ip_address, user_agent)

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
        """토큰 검증 (지속된 deny-list 를 무효화의 근거로 사용)"""
        with self._lock:
            if token_string in self._deny_list:
                return None

            payload = self._jwt.verify_token(token_string)

            if not payload:
                return None

            user_id = payload.get("user_id")

            if not user_id or user_id not in self._users:
                return None

            return user_id

    def revoke_token(self, token_string: str) -> bool:
        """JWT 접근 토큰 또는 API 키를 명시적으로 무효화합니다."""
        with self._lock:
            if token_string in self._deny_list:
                return True
            if self._jwt.verify_token(token_string) is not None:
                self._deny_list.add(token_string)
                self._save_state()
                return True
            return self.revoke_api_key(token_string)

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

            self._save_state()
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
            self._save_state()
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
                self._save_state()
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
        """로그아웃 (세션 비활성화 + 연결된 접근 토큰 무효화)"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.is_active = False

            linked = self._tokens.get(session.token_id)
            if linked is not None:
                linked.is_valid = False
                if linked.token_string:
                    self._deny_list.add(linked.token_string)

            self._save_state()
            return True

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

            self._save_state()
            return True

    def upsert_admin_credentials(
        self, user_id: str, username: str, password: str
    ) -> bool:
        """관리자 계정 생성/비밀번호 갱신 (부트스트랩 전용, 멱등)."""
        with self._lock:
            password_hash, salt = PasswordHasher.hash_password(password)
            self._users[user_id] = {
                "username": username,
                "password_hash": password_hash,
                "salt": salt,
                "role": "admin",
                "created_at": time.time(),
                "last_login": None,
                "is_active": True,
            }
            self._save_state()
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
