"""
DiskCache pickle RCE 보안 회귀 테스트.

목적: pickle 역직렬화 제거 후 다음을 보장한다.
- 악성 pickle 바이트가 .cache 파일에 있어도 절대 실행되지 않음 (RCE 차단)
- 레거시 pickle 포맷 파일은 신뢰하지 않고 폐기됨
- JSON 라운드트립은 정상 동작
- 활성 HMAC 하에서 .meta 변조는 무결성 검증 실패로 거부됨
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from services.optimization.caching_optimizer import DiskCache  # noqa: E402


@pytest.fixture
def sentinel_path(tmp_path):
    """pickle 페이로드가 실행되면 생성되는 센티넬 파일 경로."""
    return tmp_path / "PWNED_SENTINEL"


def _write_malicious_pickle(cache_dir: Path, key: str, sentinel: Path) -> None:
    """디스크에 악성 pickle 바이트를 직접 기록 (verification 우회 시도)."""
    import hashlib

    payload = (
        b"(c__builtin__\nopen\n"
        b"(S'" + sentinel.as_posix().encode() + b"'\nS'w'\n"
        b"tR(S'evil'\nS'w'\ntR."
    )
    cache_file = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.cache"
    cache_file.write_bytes(payload)


@pytest.mark.asyncio
async def test_malicious_pickle_never_executed(tmp_path, monkeypatch, sentinel_path):
    """악성 pickle 파일이 실행되지 않고 폐기되는지 확인 (RCE 차단)."""
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    # .meta 가 없으면 full_verification 이 메타데이터 누락으로 실패 -> 폐기
    mock_manager.full_verification.return_value = (False, "메타데이터 없음")
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    cache = DiskCache(cache_dir=str(tmp_path))
    key = "attack"
    _write_malicious_pickle(Path(tmp_path), key, sentinel_path)

    result = await cache.get(key)

    assert result is None
    assert not sentinel_path.exists(), "악성 pickle 이 실행되었다 (RCE 발생!)"
    # 캐시 파일은 폐기되어야 함
    assert list(Path(tmp_path).glob("*.cache")) == []


@pytest.mark.asyncio
async def test_legacy_pickle_discarded(tmp_path, monkeypatch):
    """레거시 pickle 포맷 파일은 신뢰하지 않고 폐기되는지 확인."""
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (True, None)
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    cache = DiskCache(cache_dir=str(tmp_path))
    key = "legacy"
    legacy_obj = {
        "_fmt": "json-v2",
        "key": key,
        "value": "v",
        "created_at": 1.0,
        "accessed_at": 1.0,
        "ttl_seconds": 86400.0,
        "hit_count": 0,
        "metadata": {},
    }
    cache_file = (
        Path(tmp_path)
        / f"{__import__('hashlib').sha256(key.encode()).hexdigest()}.cache"
    )
    cache_file.write_bytes(pickle.dumps(legacy_obj))

    result = await cache.get(key)

    assert result is None
    assert list(Path(tmp_path).glob("*.cache")) == []


@pytest.mark.asyncio
async def test_json_roundtrip_str_dict_int(tmp_path, monkeypatch):
    """JSON 라운드트립이 다양한 값에서 정상 동작하는지 확인."""
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (True, None)
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    cache = DiskCache(cache_dir=str(tmp_path))

    for key, value in [
        ("k_str", "hello"),
        ("k_int", 42),
        ("k_dict", {"answer": "ok", "n": [1, 2, 3]}),
    ]:
        await cache.set(key, value)
        assert await cache.get(key) == value

    # on-disk 파일이 유효한 JSON 인지 확인
    for cache_file in Path(tmp_path).glob("*.cache"):
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        assert raw.get("_fmt") == "json-v2"


def test_hmac_tamper_rejected():
    """활성 HMAC 하에서 .meta 무결성 해시 변조가 거부되는지 확인."""
    from security.cache_security import (
        CacheIntegrityError,
        CacheSecurityManager,
    )

    with tempfile.TemporaryDirectory() as tmp:
        secret = "this_is_a_very_secret_key_at_least_32_chars"
        mgr = CacheSecurityManager(security_level="high", hmac_secret=secret)

        cache_file = Path(tmp) / "entry.cache"
        cache_file.write_text(
            '{"_fmt": "json-v2", "key": "k", "value": "v"}', encoding="utf-8"
        )

        metadata = mgr.create_metadata_for_file(str(cache_file))

        # .meta 무결성 해시를 변조
        tampered = metadata.model_dump()
        tampered["integrity_hmac"] = "deadbeef" * 8
        meta_path = str(cache_file) + ".meta"
        import json as _json

        Path(meta_path).write_text(
            _json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
        )

        with pytest.raises(CacheIntegrityError):
            mgr.verify_cache_integrity(str(cache_file), metadata_path=meta_path)
