# src/ui/session_sync.py
"""
UI-sync adapter that mirrors ``SessionManager`` state into Streamlit.

This module lives in ``src/ui`` (not ``src/core``) so the core session layer
stays free of any Streamlit import. ``SessionManager`` calls ``write``/``read``
on whatever adapter is installed via ``SessionManager.set_ui_sync``; when none
is installed it falls back to the global store only (see core.session.manager).

The ``write`` semantics intentionally reproduce the previous in-core behavior
(manager.py lines ~240-245): lists are sliced and dicts are ``copy.copy``'d
before assignment so the UI never aliases the internal mutable state.
"""

from __future__ import annotations

import copy
from typing import Any

import streamlit as st


def _copy_plain(value: Any) -> Any:
    """Plain 컨테이너만 재귀 복사하고 불투명 객체(엔진/락 등)는 참조 유지."""
    if type(value) is dict:
        return {k: _copy_plain(v) for k, v in value.items()}
    if type(value) is list:
        return [_copy_plain(v) for v in value]
    if type(value) is tuple:
        return tuple(_copy_plain(v) for v in value)
    if type(value) in (set, frozenset):
        return type(value)(_copy_plain(v) for v in value)
    # dict 서브클래스(FriendlyError 등)는 재구성 시 동작이 깨지므로 공유한다.
    return value


class StreamlitSessionSync:
    """Pluggable adapter routing SessionManager state to ``st.session_state``."""

    @staticmethod
    def write(key: str, val: Any) -> None:
        """Mirror ``val`` into ``st.session_state[key]``.

        D3: plain 컨테이너는 재귀 복사해 UI가 중첩 가변 상태(메시지 원소
        dict 등)를 별칭하지 않게 한다. 이전 slice/``copy.copy``는 원소 dict를
        공유해 백그라운드 ``add_message``의 제자리 ``update``와 UI 순회가
        경합했다. 엔진 같은 불투명 객체는 참조 유지(정체성 보존).
        """
        if isinstance(val, (dict, list, tuple, set, frozenset)):
            try:
                st.session_state[key] = _copy_plain(val)
                return
            except (TypeError, copy.Error):
                pass
        if isinstance(val, list):
            st.session_state[key] = val[:]
        elif isinstance(val, dict):
            st.session_state[key] = copy.copy(val)
        else:
            st.session_state[key] = val

    @staticmethod
    def read(key: str, default: Any = None) -> Any:
        """Read ``key`` from ``st.session_state`` falling back to ``default``."""
        return st.session_state.get(key, default)

    @staticmethod
    def delete(key: str) -> None:
        """Remove ``key`` from ``st.session_state`` (delete propagation).

        Absent keys are ignored so tombstone replay is idempotent.
        """
        st.session_state.pop(key, None)
