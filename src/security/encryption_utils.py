"""
Task 22-3: Encryption Utilities
데이터 암호화 및 보호 유틸리티
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import os
import json
import base64
import hashlib
import hmac


class EncryptionMethod(Enum):
    """암호화 방법"""

    AES_256_CBC = "aes_256_cbc"
    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA = "rsa"


@dataclass
class EncryptedData:
    """암호화된 데이터"""

    ciphertext: str
    iv: str  # Initialization Vector
    salt: str
    method: EncryptionMethod
    timestamp: float


class SimpleAESEncryptor:
    """간단한 AES 암호화 (모의 구현)"""

    def __init__(self, key: Optional[str] = None):
        """
        초기화
        실제 환경에서는 cryptography 라이브러리 사용
        """
        if not key:
            key = os.environ.get("ENCRYPTION_KEY", "default-secret-key-256bits!")

        # 키를 32바이트로 정규화
        key_bytes = hashlib.sha256(key.encode()).digest()
        self.key = base64.b64encode(key_bytes).decode()

    def encrypt(self, plaintext: str) -> EncryptedData:
        """데이터 암호화"""
        import time

        # 모의 구현: 간단한 XOR 암호화 (실제로는 AES 사용)
        iv = os.urandom(16).hex()[:32]
        salt = os.urandom(16).hex()[:32]

        # 간단한 변환 (실제로는 AES-256-CBC 사용)
        plaintext_bytes = plaintext.encode("utf-8")
        key_bytes = base64.b64decode(self.key)

        # XOR 기반 간단한 암호화
        ciphertext_bytes = bytearray()
        for i, byte in enumerate(plaintext_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            ciphertext_bytes.append(byte ^ key_byte)

        ciphertext = base64.b64encode(ciphertext_bytes).decode()

        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv,
            salt=salt,
            method=EncryptionMethod.AES_256_CBC,
            timestamp=time.time(),
        )

    def decrypt(self, encrypted_data: EncryptedData) -> Optional[str]:
        """데이터 복호화"""
        try:
            ciphertext_bytes = base64.b64decode(encrypted_data.ciphertext)
            key_bytes = base64.b64decode(self.key)

            # XOR 기반 간단한 복호화
            plaintext_bytes = bytearray()
            for i, byte in enumerate(ciphertext_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                plaintext_bytes.append(byte ^ key_byte)

            return plaintext_bytes.decode("utf-8")

        except Exception:
            return None


class DataEncryptor:
    """데이터 암호화 관리자"""

    def __init__(self, encryption_key: Optional[str] = None):
        self._aes_encryptor = SimpleAESEncryptor(encryption_key)
        self._key = encryption_key or "default-key"

    def encrypt_string(self, plaintext: str) -> str:
        """문자열 암호화"""
        encrypted_data = self._aes_encryptor.encrypt(plaintext)

        # JSON으로 직렬화
        payload = {
            "ciphertext": encrypted_data.ciphertext,
            "iv": encrypted_data.iv,
            "salt": encrypted_data.salt,
            "method": encrypted_data.method.value,
            "timestamp": encrypted_data.timestamp,
        }

        return base64.b64encode(json.dumps(payload).encode()).decode()

    def decrypt_string(self, encrypted_string: str) -> Optional[str]:
        """문자열 복호화"""
        try:
            payload = json.loads(base64.b64decode(encrypted_string).decode())

            encrypted_data = EncryptedData(
                ciphertext=payload["ciphertext"],
                iv=payload["iv"],
                salt=payload["salt"],
                method=EncryptionMethod(payload["method"]),
                timestamp=payload["timestamp"],
            )

            return self._aes_encryptor.decrypt(encrypted_data)

        except Exception:
            return None

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """딕셔너리 암호화"""
        json_str = json.dumps(data)
        return self.encrypt_string(json_str)

    def decrypt_dict(self, encrypted_string: str) -> Optional[Dict[str, Any]]:
        """딕셔너리 복호화"""
        decrypted = self.decrypt_string(encrypted_string)

        if not decrypted:
            return None

        try:
            return json.loads(decrypted)
        except Exception:
            return None

    def hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """비밀번호 검증"""
        return hmac.compare_digest(self.hash_password(password), password_hash)

    def generate_token(self, length: int = 32) -> str:
        """토큰 생성"""
        return base64.urlsafe_b64encode(os.urandom(length)).decode().rstrip("=")

    def hash_data(self, data: str) -> str:
        """데이터 해싱"""
        return hashlib.sha256(data.encode()).hexdigest()


class EncryptionManager:
    """암호화 관리자"""

    def __init__(self):
        self._default_encryptor = DataEncryptor()
        self._field_configs: Dict[str, Dict[str, Any]] = {}

    def configure_field_encryption(
        self,
        field_name: str,
        should_encrypt: bool = True,
        method: EncryptionMethod = EncryptionMethod.AES_256_CBC,
    ):
        """필드 암호화 설정"""
        self._field_configs[field_name] = {"encrypt": should_encrypt, "method": method}

    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """민감한 데이터 암호화"""
        encrypted_data = data.copy()

        # 민감한 필드 목록
        sensitive_fields = [
            "password",
            "email",
            "phone",
            "ssn",
            "credit_card",
            "bank_account",
            "api_key",
        ]

        for key, value in encrypted_data.items():
            if key in sensitive_fields or key in self._field_configs:
                if isinstance(value, str):
                    config = self._field_configs.get(key, {"encrypt": True})

                    if config.get("encrypt"):
                        encrypted_data[key] = {
                            "encrypted": True,
                            "value": self._default_encryptor.encrypt_string(value),
                        }

        return encrypted_data

    def decrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """민감한 데이터 복호화"""
        decrypted_data = data.copy()

        for key, value in decrypted_data.items():
            if isinstance(value, dict) and value.get("encrypted"):
                decrypted_value = self._default_encryptor.decrypt_string(
                    value.get("value", "")
                )

                if decrypted_value:
                    decrypted_data[key] = decrypted_value

        return decrypted_data

    def encrypt_pii(self, pii: str, method: str = "mask") -> str:
        """개인식별정보(PII) 암호화"""
        if method == "mask":
            # 마스킹
            if len(pii) <= 4:
                return "*" * len(pii)
            return pii[:2] + "*" * (len(pii) - 4) + pii[-2:]
        elif method == "encrypt":
            # 암호화
            return self._default_encryptor.encrypt_string(pii)
        else:
            return pii

    def secure_delete(self, sensitive_string: str) -> bool:
        """보안 삭제"""
        # 민감한 데이터를 메모리에서 안전하게 제거
        try:
            # 메모리 덮어쓰기 (모의 구현)
            import ctypes

            ctypes.memset(id(sensitive_string), 0, len(sensitive_string))
            return True
        except Exception:
            return False

    def get_encryption_status(self) -> Dict[str, Any]:
        """암호화 상태"""
        return {
            "encryption_enabled": True,
            "default_method": EncryptionMethod.AES_256_CBC.value,
            "configured_fields": len(self._field_configs),
            "sensitive_fields": [
                "password",
                "email",
                "phone",
                "ssn",
                "credit_card",
                "bank_account",
                "api_key",
            ],
        }


class SecureDataStorage:
    """보안 데이터 저장소"""

    def __init__(self):
        self._encryption_manager = EncryptionManager()
        self._data_store: Dict[str, Any] = {}

    def store(self, key: str, value: Any, sensitive: bool = False) -> bool:
        """데이터 저장"""
        try:
            if sensitive:
                if isinstance(value, dict):
                    encrypted_value = self._encryption_manager.encrypt_sensitive_data(
                        value
                    )
                elif isinstance(value, str):
                    encrypted_value = (
                        self._encryption_manager._default_encryptor.encrypt_string(
                            value
                        )
                    )
                else:
                    encrypted_value = value

                self._data_store[key] = {"encrypted": True, "value": encrypted_value}
            else:
                self._data_store[key] = {"encrypted": False, "value": value}

            return True
        except Exception:
            return False

    def retrieve(self, key: str) -> Optional[Any]:
        """데이터 조회"""
        if key not in self._data_store:
            return None

        stored = self._data_store[key]

        if stored.get("encrypted"):
            if isinstance(stored["value"], dict):
                return self._encryption_manager.decrypt_sensitive_data(stored["value"])
            else:
                return self._encryption_manager._default_encryptor.decrypt_string(
                    stored["value"]
                )

        return stored["value"]

    def delete(self, key: str) -> bool:
        """데이터 삭제"""
        if key in self._data_store:
            del self._data_store[key]
            return True
        return False
