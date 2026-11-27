"""UI-owned runtime settings keep config.yml out of the live value path."""

from __future__ import annotations

import pytest

import common.config as config_mod
import core.pipeline_builder as pipeline_mod
from common.config import (
    ENABLE_VECTOR_CACHE,
    GRADING_ENABLED,
    QUERY_CACHE_ENABLED,
    RERANKER_ENGINE,
)
from common.runtime_settings import RuntimeSetting


def test_defaults_match_removed_config_keys() -> None:
    """Removing the YAML keys must not change startup behavior."""
    assert bool(ENABLE_VECTOR_CACHE) is True
    assert bool(QUERY_CACHE_ENABLED) is True
    assert bool(GRADING_ENABLED) is True
    assert RERANKER_ENGINE == "auto"
    assert "global_cache" not in config_mod._load_config()
    assert "enabled" not in config_mod._load_config()["rag"]["query_cache"]
    assert "grading" not in config_mod._load_config()["rag"]


def test_override_is_visible_to_consumers_and_resettable() -> None:
    try:
        ENABLE_VECTOR_CACHE.set(False)
        assert bool(ENABLE_VECTOR_CACHE) is False
        assert bool(pipeline_mod.ENABLE_VECTOR_CACHE) is False
    finally:
        ENABLE_VECTOR_CACHE.reset()
    assert bool(pipeline_mod.ENABLE_VECTOR_CACHE) is True


def test_engine_choice_validated() -> None:
    try:
        RERANKER_ENGINE.set("semantic")
        assert RERANKER_ENGINE.lower() == "semantic"
        with pytest.raises(ValueError, match="reranker_engine"):
            RERANKER_ENGINE.set("bogus")
    finally:
        RERANKER_ENGINE.reset()


def test_runtime_setting_is_not_a_plain_bool() -> None:
    assert isinstance(GRADING_ENABLED, RuntimeSetting)
    assert bool(GRADING_ENABLED)
