"""st.status.__exit__의 실제 강제-완료 동작을 재현하는 라이프사이클 회귀 테스트."""

from __future__ import annotations

from unittest.mock import patch

import ui.components.doc_analysis.panel as panel_mod
from core.build_state import BuildPhase, initial_build_state
from core.session import SessionManager


class RealStreamlitStatusMock:
    """Streamlit StatusContainer.__exit__의 실제 라이프사이클을 재현하는 Mock."""

    def __init__(self, label, expanded=True, state="running"):
        self.label = label
        self.expanded = expanded
        self.state = state
        self.history = []

    def update(self, label=None, expanded=None, state=None):
        if label is not None:
            self.label = label
        if expanded is not None:
            self.expanded = expanded
        if state is not None:
            self.state = state
        self.history.append({"state": self.state, "expanded": self.expanded})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Streamlit 실제 소스코드: running 상태면 무조건 complete로 자동 전환
        if self.state == "running":
            self.update(state="complete", expanded=False)
        return False


def test_render_document_panel_maintains_running_state_after_exit():
    sid = "test_panel_status_exit_sid"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set_session_id(sid)

    build = initial_build_state("doc.pdf", "hash_test")
    build.phase = BuildPhase.EXTRACTING
    build.pct = 30
    SessionManager.set("build_state", build, session_id=sid)

    container_instance = None

    def fake_status(label, expanded=True, state="running"):
        nonlocal container_instance
        container_instance = RealStreamlitStatusMock(label, expanded, state)
        return container_instance

    with (
        patch.object(panel_mod.st, "status", side_effect=fake_status),
        patch.object(panel_mod.st, "progress"),
        patch.object(panel_mod.st, "caption"),
        patch.object(panel_mod.st, "button"),
    ):
        panel_mod.render_document_panel(sid)

    # [GREEN 조건] with 문이 완전히 종료된 후 최종 state가 running이어야 함
    assert container_instance is not None
    assert container_instance.state == "running", (
        f"__exit__에 의해 상태가 강제 종료됨: 최종 상태={container_instance.state}"
    )
    assert container_instance.expanded is True
