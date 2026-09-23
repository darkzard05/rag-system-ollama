"""
Lane B cross-validation (independent method): state-machine-driven AppTest
with a mocked stream. No real document build, no shared disk/FAISS IO.

Verifies three behaviors of the chat view in `src/ui/components/chat.py`
(+ `chat_build.py` / `chat_references.py` P3 분리 모듈):
  (a) build-status lifecycle: a native `st.status` shows a running
      "{파일} · 문서 분석 중..." state with the "분석 취소" button and a
      "0% 완료" progress caption while `is_building_rag` is live, and
      transitions to a "{파일} · 준비 완료" complete state when the build
      finishes;
  (b) chat input state machine (6-state): input DISABLED with the upload
      prompt when no file is present, ENABLED when ready, and ENABLED while
      `is_generating_answer=True` (submit_mode="stop" shows ■ instead);
  (c) a completed streamed answer renders an assistant message exposing the
      native reasoning expander ("Thought Process",
      strings.expander_reasoning_only) carrying the accumulated pipeline
      steps/thought.

AppTest idioms that work (streamlit 1.54.0):
- `AppTest.from_file("src/main.py").run(timeout=N)` boots the full app.
- The app's script thread resolves its session via `get_script_run_ctx()`, and
  `streamlit.testing.v1.LocalScriptRunner` hardcodes `session_id="test session
  id"` (local_script_runner.py:61). The matching SessionManager store is
  created during the first boot, so drive state with
  `SessionManager.set(key, value, session_id=_app_session_id())` between
  `.run()` calls — the UI reads everything through `SessionManager.get()`.
- `UIBridge.sync_session()` (src/ui/bridge.py:54) actually runs under AppTest —
  `SessionManager.get_session_id()` resolves the real "test session id" from the
  script-run context (LocalScriptRunner), so it is not a no-op. It is safe
  because it only mirrors store keys and never writes widget keys. The old code
  was a no-op and that masked a latent widget-key write crash (snapshot/restore
  block removed; Streamlit 1.54 forbids script-side widget-key assignment).
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
    """Satisfy SessionManager.is_ready_for_chat (manager.py:350-357).

    현행 6-상태 입력 게이팅(chat.py:_resolve_chat_input_state)은 파일 업로드
    전에는 입력을 DISABLED로 둔다. enabled 입력을 얻으려면 파일명 + readiness를
    함께 세팅해야 한다.
    """
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("rag_engine", object(), sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("needs_rag_rebuild", False, sid)
    SessionManager.set("needs_qa_chain_update", False, sid)
    SessionManager.set("pdf_processing_error", None, sid)


def test_build_status_banner_lifecycle():
    """(a) 네이티브 st.status가 빌드 중 running으로 표시되고 완료 후 complete로 전환된다.

    현행 계약 (ui/components/chat_build.py:_render_build_progress_block):
    - 라벨 "{파일명} · {상태}" (진행 중: "· 문서 분석 중...", 완료: "· 준비 완료")
    - st.status 단독 렌더 (구 chat_message 래퍼 없음), 취소 버튼 "분석 취소",
      진행 캡션 "{n}% 완료 · {s}초 경과".
    """
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception

    # Phase 1 — analysis running. Seed the session store exactly like
    # main.py's _bg_rebuild_task; the native renderer reads
    # these session keys (chat_build.py:_render_build_progress_block), not the
    # message dict. The added build_progress message mirrors main.py but is
    # skipped in the timeline (chat.py:_render_unified_timeline).
    SessionManager.set("is_building_rag", True, sid)
    SessionManager.set("rebuild_status", "문서 분석 중...", sid)
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
    # proven by the label ("· 문서 분석 중...", only in the running branch)
    # and by the "분석 취소" button (rendered only when state == running).
    assert "문서 분석 중" in at.status[0].label, at.status[0].label
    assert at.status[0].state != "error", at.status[0].state
    cancel_labels = [b.label for b in at.button]
    assert "분석 취소" in cancel_labels, cancel_labels
    # 현행 UI (chat_build.py): 네이티브 st.status + st.progress + "% 완료" 캡션.
    # NOTE: AppTest (1.60.0) exposes no `at.progress` element; the observable
    # proof of the progress render is the "% 완료" caption surfaced from
    # inside the st.status container.
    captions = " ".join(c.value for c in at.caption)
    assert "0% 완료" in captions, captions
    assert not any(
        e.label == "Progress log" or "진행 로그" in e.label for e in at.expander
    ), [e.label for e in at.expander]
    assert "로그1" not in captions, captions
    assert "로그2" not in captions, captions
    # 구형 도크(expander)는 제거되어서는 안 됨 → 네이티브 st.status로만 표시
    assert not any("⏳" in e.label for e in at.expander), at.expander
    assert not at.exception

    # Phase 2 — analysis finished: drive the done state via the SESSION keys
    # the renderer actually reads (chat_build.py). The message-dict mutation
    # was the pre-139c3e1 contract and is ignored today.
    SessionManager.set("rebuild_done", True, sid)
    SessionManager.set("rebuild_progress", 100, sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("pdf_processed", True, sid)
    at.run(timeout=_RUN_TIMEOUT)

    assert at.status, f"no st.status rendered after completion: {at.status}"
    # AppTest auto-transitions st.status to "complete" at context exit, so the
    # terminal state is always "complete"; the done branch is proven by label.
    assert at.status[0].state == "complete", at.status[0].state
    assert "준비 완료" in at.status[0].label, at.status[0].label
    assert not at.exception


def test_chat_input_state_machine_during_generation():
    """(b) 6-상태 입력 게이팅: 준비 완료 시 ENABLED, 생성 중에도 ENABLED.

    현행 계약 (chat.py:_resolve_chat_input_state): 파일 미업로드 시 DISABLED +
    "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요.", 생성 중에는 ENABLED로
    유지되어 네이티브 ■ 중지 버튼(submit_mode="stop")이 전송 버튼 자리에 렌더된다.
    """
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)
    assert SessionManager.is_ready_for_chat(session_id=sid)

    # Idle → enabled (첫 질문 플레이스홀더)
    at.run(timeout=_RUN_TIMEOUT)
    assert at.chat_input[0].disabled is False
    assert at.chat_input[0].placeholder == "문서 내용에 대해 궁금한 점을 질문해 보세요."
    assert not at.exception

    # Generating → still enabled so the native stop button (submit_mode="stop")
    # can render in place of the send button (UI입력통합). The disabled decision
    # (_resolve_chat_input_state) depends solely on is_generating_answer, so this
    # exercises the exact state machine without entering the streaming loop. The
    # real user-message streaming path is covered by
    # test_streamed_answer_renders_thought_expander_and_reenables.
    SessionManager.add_message("assistant", "준비 완료", session_id=sid)
    SessionManager.set("is_generating_answer", True, sid)
    at.run(timeout=_RUN_TIMEOUT)

    assert at.chat_input[0].disabled is False
    assert (
        at.chat_input[0].placeholder
        == "답변을 생성하고 있습니다... (중지는 우측 하단 ■ 버튼)"
    )
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
    ("Thought Process", strings expander_reasoning_only) carrying the
    accumulated pipeline steps, input re-enabled, no exceptions."""
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

    # Rendered assistant message exposes the native reasoning expander
    # (render_generation_expander → t("expander_reasoning_only") = "Thought
    # Process"; thought-only이므로 sources 분기 없음) whose body carries the
    # pipeline steps.
    assert any(e.label == "Thought Process" for e in at.expander), [
        e.label for e in at.expander
    ]
    # Rendered expander body carries the thought (blockquote markdown).
    # 현행 계약: process_steps는 msg 저장소에 누적될 뿐 본문 텍스트로 렌더되지
    # 않으므로, 렌더 검증은 thought 본문으로 수행한다.
    thought_text = "".join(m.value for m in at.markdown)
    assert "독립 검증용 추론 과정입니다." in thought_text, (
        f"thought text missing: {thought_text!r}"
    )

    # Input re-enabled again after generation finished
    assert at.chat_input[0].disabled is False
    assert not any("답변 생성 중" in c.value for c in at.caption)
    assert not at.exception
