"""UX-7: LLM selector shows raw ID plus a role badge (stored value stays raw)."""

from __future__ import annotations

import ui.strings as strings_mod
from ui.components.sidebar import format_llm_option


def test_small_model_gets_fast_badge_ko() -> None:
    original = strings_mod.LANG
    strings_mod.set_lang("ko")
    try:
        assert (
            format_llm_option("qwen3:4b-instruct-2507-q4_K_M")
            == "qwen3:4b-instruct-2507-q4_K_M · 4B · 경량 · 빠름"
        )
    finally:
        strings_mod.set_lang(original)


def test_large_model_gets_standard_badge() -> None:
    original = strings_mod.LANG
    strings_mod.set_lang("ko")
    try:
        assert format_llm_option("llama3.3:70b") == "llama3.3:70b · 70B · 고성능"
    finally:
        strings_mod.set_lang(original)


def test_badge_follows_active_locale() -> None:
    original = strings_mod.LANG
    strings_mod.set_lang("en")
    try:
        label = format_llm_option("qwen3:4b-instruct-2507-q4_K_M")
        assert "lightweight" in label
        assert "경량" not in label
    finally:
        strings_mod.set_lang(original)


def test_id_without_size_token_renders_raw() -> None:
    assert format_llm_option("phi4:latest") == "phi4:latest"
