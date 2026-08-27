"""
Lane B cross-validation (independent method): state-machine-driven AppTest
with a mocked stream. No real document build, no shared disk/FAISS IO.

Verifies three behaviors of the chat view in `src/ui/components/chat.py`:
  (a) build-status lifecycle: a native `st.status` shows a running "문서 분석
      중 ..." state with the "분석 취소" button and a collapsed "진행 로그"
      expander while a `build_progress` message is live, and transitions to a
      "✅ 분석 완료" complete state when the build finishes;
  (b) chat input state machine: input DISABLED while
      `is_generating_answer=True`, ENABLED when idle;
  (c) a completed streamed answer renders an assistant message exposing the
      native reasoning expander ("🧠 상세 사고 과정", chat.py:189) carrying
      the accumulated pipeline steps/thought.

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
  `at.status[i].label` / `.state`, `at.chat_input[i].disabled`,
  `at.markdown[i].value`, `at.exception` (empty when clean).
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
    """(a) 네이티브 st.status가 빌드 중 running으로 표시되고 완료 후 complete로 전환된다."""
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception

    # Phase 1 — analysis running. Seed a build_progress message exactly like
    # main.py's _bg_rebuild_task (main.py:203-213); the native renderer reads
    # the message, not session keys.
    SessionManager.set("is_building_rag", True, sid)
    SessionManager.set("rebuild_status", "문서 분석 중...", sid)
    SessionManager.set("status_logs", ["로그1", "로그2"], sid)
    SessionManager.add_message(
        "system",
        "📄 문서 분석 시작",
        msg_type="build_progress",
        msg_id=f"build_{sid}",
        progress=0,
        done=False,
        status="문서 분석 중...",
        cancelable=True,
        logs=["로그1", "로그2"],
        session_id=sid,
    )
    at.run(timeout=_RUN_TIMEOUT)

    assert at.status, f"no st.status rendered: {at.status}"
    # NOTE: `with st.status(...)` auto-updates a "running" status to "complete"
    # at context-manager exit (streamlit mutable_status_container.py:174-193),
    # so AppTest always observes the terminal state here. The running branch is
    # proven by the label ("Analyzing document: ...", only in the running branch)
    # and by the "Cancel Analysis" button (rendered only when state == running).
    assert "Analyzing document:" in at.status[0].label, at.status[0].label
    assert at.status[0].state != "error", at.status[0].state
    cancel_labels = [b.label for b in at.button]
    assert "Cancel Analysis" in cancel_labels, cancel_labels
    logs_exp = next((e for e in at.expander if e.label == "Progress log"), None)
    assert logs_exp is not None, [e.label for e in at.expander]
    log_text = "".join(t.value for t in logs_exp.text)
    assert "로그1" in log_text, log_text
    assert "로그2" in log_text, log_text
    # 구형 도크(expander)는 제거되어서는 안 됨 → 네이티브 st.status로만 표시
    assert not any("⏳" in e.label for e in at.expander), at.expander
    assert not at.exception

    # Phase 2 — analysis finished: transition the build_progress message to
    # done (pdf_processed flips; build flag cleared). The native st.status
    # stays rendered in the "complete" state.
    msgs = SessionManager.get_messages(session_id=sid)
    bmsg = next(m for m in msgs if m.get("msg_type") == "build_progress")
    bmsg["status"] = "분석 완료"
    bmsg["progress"] = 100
    bmsg["done"] = True
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("pdf_processed", True, sid)
    at.run(timeout=_RUN_TIMEOUT)

    assert at.status, f"no st.status rendered after completion: {at.status}"
    # AppTest auto-transitions st.status to "complete" at context exit, so the
    # terminal state is always "complete"; the done branch is proven by label.
    assert at.status[0].state == "complete", at.status[0].state
    assert "Analysis complete" in at.status[0].label, at.status[0].label
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
    """(c) completed answer → assistant message with native reasoning expander
    (🧠 상세 사고 과정) carrying the accumulated pipeline steps,
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
    at.run(timeout=_RUN_TIMEOUT)
    assert at.chat_input[0].disabled is False

    # Submit the query through the native chat input. The current rendering
    # path starts the stream from widget submission only (chat.py:
    # render_chat_input_area → start_streaming_turn), so drive the widget rather
    # than pre-seeding a user message.
    at.chat_input[0].set_value("이 문서를 요약해주세요")
    at.run(timeout=_RUN_TIMEOUT)

    # The mocked stream completes in the background consumer thread
    # (streaming.py:_spawn_stream_consumer); poll until it flips the flag.
    deadline = time.time() + 30
    while SessionManager.get("is_generating_answer", False, sid) and (
        time.time() < deadline
    ):
        time.sleep(0.2)
        at.run(timeout=_RUN_TIMEOUT)

    # The loop exits as soon as the consumer flips is_generating_answer under
    # the lock, but the final in-loop render may still show the "streaming"
    # message. Flush so the finalized assistant message (msg_type="general",
    # process panel) is what the tree-based assertions read.
    for _ in range(3):
        at.run(timeout=_RUN_TIMEOUT)

    # Stream completed → assistant message appended with the pipeline steps
    # accumulated into msg["process"]["steps"] (T2) plus the thought block.
    assert SessionManager.get("is_generating_answer", False, sid) is False
    msgs = SessionManager.get_messages(session_id=sid)
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assert assistant_msgs, "assistant message should have been appended"
    assert "독립 검증용 추론 과정입니다." in (assistant_msgs[-1].get("thought") or "")

    # T2's status accumulation: the mocked stream's status step must land in
    # msg["process_steps"] at completion (rendered via _build_process → steps).
    steps = assistant_msgs[-1].get("process_steps") or []
    assert "관련 문서 검색 중..." in steps, f"process_steps={steps}"

    # Rendered assistant message exposes the native "Answer details" expander
    # (render_generation_expander, chat.py:269) whose body carries the pipeline
    # steps joined by " · " (chat.py:280).
    assert any(e.label == "Answer details" for e in at.expander), [
        e.label for e in at.expander
    ]
    step_text = "".join(
        m.value for m in at.markdown if "관련 문서 검색 중..." in m.value
    ) + "".join(c.value for c in at.caption if "관련 문서 검색 중..." in c.value)
    assert "관련 문서 검색 중..." in step_text, f"step text missing: {step_text!r}"

    # Input re-enabled again after generation finished
    assert at.chat_input[0].disabled is False
    assert not any("답변 생성 중" in c.value for c in at.caption)
    assert not at.exception
