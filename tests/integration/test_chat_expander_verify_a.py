"""
Lane A cross-validation: chat "expander" + interactive chat rendering.

Full-flow test: (1) PDF upload -> (2) analysis completes -> (3) question via
chat input. Verifies that the build-status expander disappears, the chat input
enables, user/assistant chat messages render, and no uncaught UI exceptions
occur.

Environment notes / documented adaptations (streamlit 1.54.0, AppTest):
- `IS_CI_TEST=true` activates the project's headless stubs
  (model_loader.py:340-344 FakeEmbeddings, :442-456 GenericFakeChatModel).
- AppTest 1.54 has NO `st.file_uploader` element class, so the upload widget
  cannot be driven with `.upload()`. The upload is therefore simulated through
  the exact state `on_file_upload()` (main.py:271-328) writes: the PDF is
  copied to `data/temp` and SessionManager keys (`pdf_file_path`,
  `last_uploaded_file_name`, `file_hash`, `new_file_uploaded`) are set, then a
  normal `at.run()` lets main.py's `_handle_pending_tasks()` (main.py:378-458)
  launch the real `_bg_rebuild_task` pipeline.
- SessionManager is a process-global store shared with the AppTest script
  thread (sid "test session id"), so build state is polled from
  `SessionManager._fallback_sessions`.
- `st.chat_input` disabled state is read via the proto fallthrough
  (`at.chat_input[i].disabled`), placeholder via `.placeholder`.

Known pipeline defects surfaced by this full-flow run (both now FIXED):
- BUG 1 (production-relevant): graph_builder.py:537 `"relevant_docs_count":
  len(docs)` raised `TypeError: object of type 'NoneType' has no len()` on the
  general-intent path (preprocess -> generate skips retrieve, so
  `relevant_docs` was never set). chat.py:606's except tuple only caught
  RuntimeError/ConnectionError/CancelledError, so the TypeError escaped and
  the turn ended with NO assistant message (silent failure for casual/greeting
  messages). FIXED via graph_builder.py `relevant_docs or []`; the
  `c1_general_assistant_message_rendered` regression assertion below guards
  this fix.
- BUG 2 (stub-specific): the headless GenericFakeChatModel
  (model_loader.py:447-456) exposed a 2-item iterator shared by the grade and
  generate nodes; grade retries exhausted it and generate then died with
  `StopIteration` -> the semantic RAG question rendered the app's formatted
  error message (this cannot happen with a real Ollama model). FIXED in the
  stub.
- NOTE: AppTest `at.exception` reflects only `st.exception()` widget calls,
  NOT every uncaught exception on the AppTest script thread; uncaught
  exceptions in the background generation thread are therefore observed
  indirectly via session state (assistant message presence and the
  `is_generating_answer` flag).
- The stub LLM never emits a "thought" (graph_builder.py:510-515 fallback), so
  the real-pipeline answer cannot carry the `<details class="thought-expander">`
  HTML; the thought expander rendering path (chat.py:213-214) is therefore
  verified with a message-store injection as the closest observable equivalent.
"""

import os
import sys
import time

import pytest
from streamlit.testing.v1 import AppTest

os.environ["IS_CI_TEST"] = "true"

sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session import SessionManager  # noqa: E402

PDF_PATH = os.path.join("tests", "data", "2201.07520v1.pdf")
THOUGHT_MARKER = '<details class="thought-expander">'
ANALYSIS_WALL_TIMEOUT = 240  # seconds
ANSWER_WALL_TIMEOUT = 300  # seconds

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def _app_session_id() -> str:
    """Session id used by the AppTest script thread (SessionManager store)."""
    fb = SessionManager._fallback_sessions
    return max(fb, key=lambda k: fb[k]["last_accessed"])


def _session(key: str, sid: str, default=None):
    state = SessionManager._fallback_sessions.get(sid, {})
    return state.get(key, default)


def _run_until_settled(at: AppTest, max_runs: int = 3) -> None:
    """Re-run a few times so pending reruns/fragments flush into the tree."""
    for _ in range(max_runs):
        at.run()
        time.sleep(1)


def _wait_for_generation_finished(at: AppTest, sid: str, timeout: int) -> bool:
    """Wait until a submitted turn is no longer generating.

    (A turn may end without an assistant message when the pipeline crashes
    with a TypeError, so completion is detected via is_generating_answer.)
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        generating = bool(_session("is_generating_answer", sid, False))
        msgs = _session("messages", sid, []) or []
        if not generating and any(m.get("role") == "user" for m in msgs):
            return True
        at.run()
        time.sleep(2)
    return False


def _rendered_messages(at: AppTest):
    return [
        {
            "name": m.name,
            "markdowns": [md.value for md in m.markdown],
        }
        for m in at.chat_message
    ]


def _submit_chat(at: AppTest, text: str) -> None:
    assert len(at.chat_input) >= 1, "no chat_input element in tree"
    at.chat_input[0].set_value(text)
    at.run()


def _chat_input_ready(at: AppTest) -> bool:
    return (
        len(at.chat_input) >= 1
        and at.chat_input[0].disabled is False
        and at.chat_input[0].placeholder == "추가 질문을 입력하세요..."
    )


def test_chat_expander_and_interactive_chat_render() -> None:
    # Fresh process-global store so other tests' state cannot leak in.
    SessionManager.reset()

    at = AppTest.from_file("src/main.py", default_timeout=300)
    at.run()
    sid = _app_session_id()
    print(f"AppTest session id: {sid}")
    record("app_initial_run", True, f"initial at.run() ok (sid={sid})")

    # ------------------------------------------------------------------
    # (1) PDF upload (simulated, see module docstring)
    # ------------------------------------------------------------------
    with open(PDF_PATH, "rb") as f:
        file_bytes = f.read()
    from core.document_processor import compute_file_hash

    file_hash = compute_file_hash("", data=file_bytes)
    tmp_path = os.path.join("data", "temp", f"upload_{sid}_{int(time.time())}.pdf")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    SessionManager.set("pdf_file_path", tmp_path, session_id=sid)
    SessionManager.set("last_uploaded_file_name", "2201.07520v1.pdf", session_id=sid)
    SessionManager.set("file_hash", file_hash, session_id=sid)
    SessionManager.set("new_file_uploaded", True, session_id=sid)
    at.run()  # main.py:_handle_pending_tasks -> real _bg_rebuild_task
    record("upload_triggered", True, f"upload simulated, build started ({tmp_path})")

    # ------------------------------------------------------------------
    # (2) Wait for analysis to complete
    # ------------------------------------------------------------------
    deadline = time.time() + ANALYSIS_WALL_TIMEOUT
    completed = False
    while time.time() < deadline:
        if _session("pdf_processed", sid) and not _session(
            "is_building_rag", sid, False
        ):
            completed = True
            break
        build_err = _session("pdf_processing_error", sid)
        if build_err:
            pytest.fail(f"analysis failed: {build_err}")
        at.run()
        time.sleep(2)
    assert completed, "analysis did not complete within wall timeout"
    _run_until_settled(at)  # render the post-build UI tree
    record(
        "analysis_completed",
        True,
        f"pdf_processed={_session('pdf_processed', sid)} "
        f"is_building_rag={_session('is_building_rag', sid, False)}",
    )

    # ASSERT (a): build-status expander is GONE after completion.
    # Idiom: iterate at.expander (list of Expander elements), read .label.
    expander_labels = [e.label for e in at.expander]
    a_ok = not any(str(lbl).startswith("⏳") for lbl in expander_labels)
    record(
        "a_build_status_expander_gone",
        a_ok,
        f"expander labels={expander_labels!r} (no '⏳' prefix expected)",
    )
    assert a_ok, f"build-status expander still present: {expander_labels!r}"

    # ASSERT (b): chat input present and ENABLED after analysis.
    b_ok = _chat_input_ready(at)
    record(
        "b_chat_input_present_and_enabled",
        b_ok,
        f"disabled={at.chat_input[0].disabled if at.chat_input else 'N/A'} "
        f"placeholder={at.chat_input[0].placeholder if at.chat_input else 'N/A'!r}",
    )
    assert b_ok, "chat input missing/disabled after analysis"

    # ------------------------------------------------------------------
    # (3a) General-intent message (fresh stub LLM): regression guard for
    #      BUG 1 -- graph_builder.py:537 raised TypeError on the
    #      general-intent path (len(docs) on None) and, because chat.py:606
    #      didn't catch TypeError, the turn ended with NO assistant message.
    #      FIXED via `relevant_docs or []`; assert the assistant message now
    #      renders. UI must stay alive.
    # ------------------------------------------------------------------
    q_general = "안녕하세요"
    _submit_chat(at, q_general)
    assert _wait_for_generation_finished(at, sid, ANSWER_WALL_TIMEOUT), (
        "general turn did not finish"
    )
    _run_until_settled(at)

    rendered = _rendered_messages(at)
    user_msgs = [m for m in rendered if m["name"] == "user"]
    asst_msgs = [m for m in rendered if m["name"] == "assistant"]
    c2_ok = any(q_general in md for m in user_msgs for md in m["markdowns"])
    record(
        "c2_general_user_message_rendered",
        c2_ok,
        f"user chat_message count={len(user_msgs)}",
    )
    assert c2_ok

    # ASSERT (c1): BUG 1 regression -- the general-intent turn must render an
    # assistant chat message with non-empty markdown (guards graph_builder.py
    # `relevant_docs or []`; previously the TypeError produced NO assistant
    # message). Mirrors the d1 assistant-message assertion idiom.
    asst_content_general = "".join(md for m in asst_msgs for md in m["markdowns"])
    c1_ok = len(asst_msgs) >= 1 and bool(asst_content_general)
    record(
        "c1_general_assistant_message_rendered",
        c1_ok,
        f"assistant chat_message count={len(asst_msgs)}, "
        f"content={asst_content_general[:140]!r}",
    )
    assert c1_ok, f"general-intent assistant message not rendered: {rendered!r}"

    e2_ok = _chat_input_ready(at)
    record("e2_chat_input_enabled_after_general", e2_ok, "")
    assert e2_ok

    f2_ok = len(at.exception) == 0
    record(
        "f2_no_uncaught_exceptions_after_general",
        f2_ok,
        f"count={len(at.exception)}",
    )
    assert f2_ok

    # ------------------------------------------------------------------
    # (3b) Mission's semantic RAG question via the chat input.
    #      With the stub LLM this renders the app's formatted error for the
    #      iterator exhaustion (BUG 2). The UI must render user + assistant
    #      messages, re-enable the input, and stay exception-free.
    # ------------------------------------------------------------------
    q = "이 논문의 핵심 내용을 요약해줘."
    _submit_chat(at, q)
    assert _wait_for_generation_finished(at, sid, ANSWER_WALL_TIMEOUT), (
        "generation did not finish within wall timeout"
    )
    _run_until_settled(at)

    rendered = _rendered_messages(at)
    user_msgs = [m for m in rendered if m["name"] == "user"]
    asst_msgs = [m for m in rendered if m["name"] == "assistant"]
    asst_content = "".join(md for m in asst_msgs for md in m["markdowns"])

    # ASSERT (c): user message rendered as a chat message.
    c_ok = any(q in md for m in user_msgs for md in m["markdowns"])
    record(
        "c_user_message_rendered",
        c_ok,
        f"user chat_message count={len(user_msgs)} (query {q!r})",
    )
    assert c_ok, f"user message not rendered: {rendered!r}"

    # ASSERT (d1): assistant message rendered in the chat window.
    d1_ok = len(asst_msgs) >= 1 and bool(asst_content)
    record(
        "d1_assistant_message_rendered",
        d1_ok,
        f"assistant chat_message count={len(asst_msgs)}, "
        f"content={asst_content[:140]!r}",
    )
    assert d1_ok, f"assistant message not rendered: {rendered!r}"

    # Document which pipeline defect caused the assistant content (if any):
    if "StopIteration" in asst_content or "알 수 없는 오류" in asst_content:
        print(
            "  [NOTE] Q assistant content is the app's formatted error for the "
            "stub LLM iterator exhaustion (BUG 2, model_loader.py:447-456 + "
            "graph_builder.py grade retries) -- UI rendered it as a chat message."
        )

    # ASSERT (e): chat input present and ENABLED again after the turn.
    e_ok = _chat_input_ready(at)
    record(
        "e_chat_input_enabled_again",
        e_ok,
        f"disabled={at.chat_input[0].disabled if at.chat_input else 'N/A'}",
    )
    assert e_ok, "chat input not re-enabled after answer"

    # ASSERT (f): no uncaught UI exceptions (AppTest .exception list).
    f_ok = len(at.exception) == 0
    record("f_no_uncaught_exceptions", f_ok, f"at.exception count={len(at.exception)}")
    assert f_ok, f"uncaught exceptions: {[e.value for e in at.exception]}"

    # ------------------------------------------------------------------
    # ASSERT (d2): thought expander HTML renders inside an assistant message.
    # Real-pipeline thought is always empty with the stub LLM
    # (graph_builder.py:510-515), so the renderer is exercised with a stored
    # message carrying a thought (closest observable equivalent).
    # ------------------------------------------------------------------
    SessionManager.add_message(
        "assistant",
        "생각 과정 검증용 답변입니다.",
        thought="테스트 생각 과정입니다.",
        session_id=sid,
    )
    _run_until_settled(at)
    rendered = _rendered_messages(at)
    thought_found = any(
        m["name"] == "assistant" and any(THOUGHT_MARKER in md for md in m["markdowns"])
        for m in rendered
    )
    thought_sample = next(
        (
            md
            for m in rendered
            if m["name"] == "assistant"
            for md in m["markdowns"]
            if THOUGHT_MARKER in md
        ),
        "",
    )
    record(
        "d2_thought_expander_html_rendered",
        thought_found,
        f'<details class="thought-expander"> found in assistant markdown '
        f"({thought_sample[:60]!r}...)",
    )
    assert thought_found, f"thought expander not rendered: {rendered!r}"

    # ------------------------------------------------------------------
    print("\n===== LANE A ASSERTION SUMMARY =====")
    for name, passed, detail in RESULTS:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    assert all(p for _, p, _ in RESULTS), "one or more Lane A assertions failed"
