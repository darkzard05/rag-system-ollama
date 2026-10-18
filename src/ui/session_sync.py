# src/ui/session_sync.py
"""
UI-sync adapter that mirrors ``SessionManager`` state into Streamlit.

This module lives in ``src/ui`` (not ``src/core``) so the core session layer
stays free of any Streamlit import. ``SessionManager`` calls ``write``/``read``
on whatever adapter is installed via ``SessionManager.set_ui_sync``; when none
is installed it falls back to the global store only (see core.session.manager).

The ``write`` semantics intentionally reproduce the previous in-core behavior
(manager.py): the top-level list is sliced and the top-level dict is copied
before assignment so the UI never aliases the internal mutable container.
Nested values are shared by reference to avoid a recursive copy on every rerun.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


class StreamlitSessionSync:
    """Pluggable adapter routing SessionManager state to ``st.session_state``."""

    @staticmethod
    def write(key: str, val: Any) -> None:
        """Mirror ``val`` into ``st.session_state[key]``.

        최상위 list/dict만 얕게 복사해 UI 미러가 저장소 컨테이너
        (``messages``/``doc_pool`` 등)에 대한 제자리 변이(``append``/``update``)를
        공유하지 않게 한다. 중첩 데이터와 불투명 객체(엔진/락 등)는 복사하지
        않아 매 rerun마다 발생하던 재귀 복사 비용을 제거한다.

        dict/list 서브클래스(``FriendlyError`` 등)는 ``type is`` 가드로 참조를
        유지한다 — 재구성 시 ``__eq__`` 등 동작이 깨지기 때문이다.
        """
        if type(val) is list:
            st.session_state[key] = val[:]
        elif type(val) is dict:
            st.session_state[key] = val.copy()
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
