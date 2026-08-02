"""
Lane B cross-validation (independent method): state-machine-driven AppTest
with a mocked stream. No real document build, no shared disk/FAISS IO.

Verifies three behaviors of the chat window in `src/ui/components/chat.py`:
  (a) build-status banner lifecycle: the building banner (st.progress + "분석
      취소" button + collapsed `<details class="build-logs">`) appears while
      `is_building_rag=True` and disappears after it flips to False;
  (b) chat input state machine: input DISABLED while
      `is_generating_answer=True`, ENABLED when idle;
  (c) a completed streamed answer renders an assistant message containing the
      thought expander HTML `<details class="thought-expander">`.

AppTest idioms that work (streamlit 1.54.0):
- `AppTest.from_file("src/main.py").run(timeout=N)` boots the full app.
- The app's script thread resolves its session via `get_script_run_ctx()`, and
  `streamlit.testing.v1.LocalScriptRunner` hardcodes `session_id="test session
  id"` (local_script_runner.py:61). The matching SessionManager store is
  created during the first boot, so drive state with
  `SessionManager.set(key, value, session_id=_app_session_id())` between
  `.run()` calls — the UI reads everything through `SessionManager.get()`.
- `UIBridge.sync_session()` (src/ui/bridge.py:54) is a no-op here because
  `ContextManager.get_current_session_id()` is never set by the app, so state
  written to the store is never clobbered by widget sync.
- A single `st.rerun()` inside the script is executed by AppTest within the
  same `.run()` call, so the post-streaming rerun settles and the returned
  tree reflects the final idle state.
- Element access: `at.expander[i].label`, `at.caption[i].value`,
  `at.chat_input[i].disabled`, `at.markdown[i].value` (unsafe HTML, incl.
  `<details class="thought-expander">`), `at.exception` (empty when clean).
"""

import asyncio
import os
import time

from streamlit.testing.v1 import AppTest

from core.rag_core import RAGSystem
from core.session import SessionManager

# Headless stub: model_loader returns fake LLM/embedders, no real Ollama calls.
os.environ.setdefault("IS_CI_TEST", "true")

# How long each script run may take before AppTest raises. Boot is ~1.5s,
# the mocked stream completes in <1s; 60s is generous headroom.
_RUN_TIMEOUT = 60


def _app_session_id() -> str:
    """Session id of the AppTest script thread (SessionManager store key).

    LocalScriptRunner hardcodes session_id="test session id"; the boot run
    creates that store. Derive it from the store to avoid hardcoding.
    """
    non_default = [k for k in SessionManager._fallback_sessions if k != "default"]
    return non_default[0] if non_default else "test session id"


def _set_ready_state(sid: str) -> None:
    """Satisfy SessionManager.is_ready_for_chat (manager.py:350-357)."""
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("rag_engine", object(), sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("needs_rag_rebuild", False, sid)
    SessionManager.set("needs_qa_chain_update", False, sid)
    SessionManager.set("pdf_processing_error", None, sid)


def test_build_status_banner_lifecycle():
    """(a) 빌드 배너(진행률+취소 버튼+접힌 로그)가 빌드 중 표시되고 완료 후 사라진다."""
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception

    # Phase 1 — analysis running
    SessionManager.set("is_building_rag", True, sid)
    SessionManager.set("rebuild_status", "문서 분석 중...", sid)
    SessionManager.set("status_logs", ["로그1", "로그2"], sid)
    at.run(timeout=_RUN_TIMEOUT)

    banner_status = [
        m.value for m in at.markdown if '<div class="build-banner-status">' in m.value
    ]
    assert banner_status, f"no build banner rendered: {[m.value for m in at.markdown]}"
    assert "⏳ 문서 분석 중..." in banner_status[0], banner_status
    cancel_labels = [b.label for b in at.button]
    assert "분석 취소" in cancel_labels, cancel_labels
    log_text = "".join(m.value for m in at.markdown if "build-log-line" in m.value)
    assert "▹ 로그1" in log_text, log_text
    assert "▹ 로그2" in log_text, log_text
    # 구형 도크(expander)는 제거되어서는 안 됨 → expander 없이 배너로만 표시
    assert not any("⏳" in e.label for e in at.expander), at.expander
    assert not at.exception

    # Phase 2 — analysis finished (pdf_processed flips; build flag cleared)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("pdf_processed", True, sid)
    at.run(timeout=_RUN_TIMEOUT)

    banner_status = [
        m.value for m in at.markdown if '<div class="build-banner-status">' in m.value
    ]
    assert not banner_status, f"build banner left: {banner_status}"
    assert not at.exception


def test_chat_input_state_machine_during_generation():
    """(b) input DISABLED while generating, ENABLED when idle."""
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)
    assert SessionManager.is_ready_for_chat(session_id=sid)

    # Idle → enabled
    at.run(timeout=_RUN_TIMEOUT)
    assert at.chat_input[0].disabled is False
    assert not at.exception

    # Generating → disabled.
    # NOTE: we use a NON-user message here deliberately. A user message would
    # enter the blocking streaming loop (chat.py:346-377) which, once the mocked
    # stream completes, resets is_generating_answer=False and calls st.rerun()
    # BEFORE render_chat_input_area() runs (ui.py renders the input after the
    # messages area) — so the post-run tree could never show the disabled input.
    # The disabled decision (_resolve_chat_input_state) depends solely on
    # is_generating_answer, so this exercises the exact state machine without
    # entering the streaming loop. The real user-message streaming path is
    # covered by test_streamed_answer_renders_thought_expander_and_reenables.
    SessionManager.add_message("assistant", "준비 완료", session_id=sid)
    SessionManager.set("is_generating_answer", True, sid)
    at.run(timeout=_RUN_TIMEOUT)

    assert at.chat_input[0].disabled is True
    assert not at.exception

    # Back to idle → enabled
    SessionManager.set("is_generating_answer", False, sid)
    at.run(timeout=_RUN_TIMEOUT)
    assert at.chat_input[0].disabled is False
    assert not at.exception


def test_streamed_answer_renders_thought_expander_and_reenables_input(
    monkeypatch,
):
    """(c) completed answer → assistant message with thought-expander HTML,
    input re-enabled, no exceptions."""
    SessionManager.reset()

    async def _mock_stream():
        # stream_graph_events consumes ("custom", dict) events (streaming_handler.py:137).
        yield ("custom", {"status": "관련 문서 검색 중..."})
        await asyncio.sleep(0.05)
        yield ("custom", {"thought": "독립 검증용 추론 과정입니다."})
        await asyncio.sleep(0.05)
        yield ("custom", {"content": "검증된 답변입니다. "})
        await asyncio.sleep(0.05)
        yield ("custom", {"content": "상세한 설명이 이어집니다."})

    async def _fake_astream(self, query, model_name=None):
        # Mirrors RAGSystem.astream (rag_core.py:138): async fn returning an
        # async generator — _sync_stream_generator awaits it (chat.py:78).
        return _mock_stream()

    monkeypatch.setattr(RAGSystem, "astream", _fake_astream)

    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)

    SessionManager.add_message("user", "이 문서를 요약해주세요", session_id=sid)
    SessionManager.set("is_generating_answer", True, sid)

    # The mocked stream completes inside this run (background thread +
    # queue, chat.py:66-141); the assistant message is appended, the flag is
    # reset and st.rerun() auto-executes, so the returned tree is the final
    # idle state. Poll as a safety net in case the rerun needs another pass.
    at.run(timeout=_RUN_TIMEOUT)
    deadline = time.time() + 15
    while SessionManager.get("is_generating_answer", False, sid) and (
        time.time() < deadline
    ):
        time.sleep(0.2)
        at.run(timeout=_RUN_TIMEOUT)

    # Stream completed → assistant message appended with the thought block
    assert SessionManager.get("is_generating_answer", False, sid) is False
    msgs = SessionManager.get_messages(session_id=sid)
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assert assistant_msgs, "assistant message should have been appended"
    assert "독립 검증용 추론 과정입니다." in (assistant_msgs[-1].get("thought") or "")

    # Rendered assistant message contains the thought expander HTML
    # (render_message → _render_thought_expander, chat.py:213-214, 178-188).
    # NOTE: main.css also contains a ".thought-expander" CSS rule, so match the
    # literal <details> tag, not just the substring "thought-expander".
    rendered_thought = [
        m.value for m in at.markdown if '<details class="thought-expander">' in m.value
    ]
    assert rendered_thought, 'no <details class="thought-expander"> rendered'

    # Input re-enabled again after generation finished
    assert at.chat_input[0].disabled is False
    assert not any("답변 생성 중" in c.value for c in at.caption)
    assert not at.exception
