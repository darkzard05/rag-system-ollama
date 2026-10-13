"""Repro: sentinel probe must call st.iframe without unsupported key (TDD Red->Green).

Official basis: st.iframe signature in Streamlit 1.60 is
iframe(src, *, width, height, tab_index) with no key parameter. Passing
key= raises TypeError, forcing the except-fallback path (double emit).
Green expectation is a single keyless call per render.
"""

import inspect

import streamlit as st

import ui.components.chat_autoscroll as sentinel_module


def test_iframe_signature_has_no_key() -> None:
    params = inspect.signature(st.iframe).parameters
    assert "key" not in params


def test_sentinel_emits_single_keyless_iframe(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_iframe(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        if "key" in kwargs:
            raise TypeError("iframe() got an unexpected keyword argument 'key'")
        return object()

    monkeypatch.setattr(sentinel_module.st, "iframe", fake_iframe)

    sentinel_module.render_autoscroll_sentinel("sid-repro")

    assert len(calls) == 1
    assert "key" not in calls[0]


def test_sentinel_payload_plants_marker() -> None:
    payload = sentinel_module.build_autoscroll_js(False)
    assert "chat-autoscroll-sentinel" in payload
