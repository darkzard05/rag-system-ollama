"""Config validation — migrated from scripts/validate_config.py.

원본 스크립트는 ``common.config._load_config()`` 로 config.yml 를 로드하여
print 로만 확인했습니다. 여기서는 로드 결과가 dict 이며 필수 최상위 키를
가지는지 assert 로 검증합니다. 라이브 모델/네트워크 의존성 없음.
"""

from common.config import CONFIG_PATH, _load_config


def test_config_file_exists() -> None:
    assert CONFIG_PATH.exists(), f"Config file not found at: {CONFIG_PATH}"


def test_load_config_returns_non_empty_dict() -> None:
    config = _load_config()

    assert isinstance(config, dict)
    assert config, "loaded config must not be empty"


def test_load_config_has_expected_top_level_keys() -> None:
    config = _load_config()

    assert "models" in config, "config must define the top-level `models` key"
    assert isinstance(config["models"], dict)
    assert "rag" in config, "config must define the top-level `rag` key"
    assert isinstance(config["rag"], dict)
