"""
Deterministic seeded AppTest repro (Q1): a streaming message rendered while
``thought=""`` keeps its body EMPTY (no spurious content), while the pipeline
step ``status`` is still surfaced in the "Answer details" expander label/body.

The current streaming render path is ``msg_type=="streaming"`` →
``_draw_streaming_message`` (chat.py:633) → ``render_generation_expander``
(chat.py:208) which draws a single native ``st.expander("Answer details")``.
This is NOT the old ``st.status`` box; the ``st.status`` widget is now used
only for the build-progress system message. The streaming renderer shows the
status step only when there is something to show (steps / thought / metrics /
documents); with ``thought=""`` and no steps the expander renders just a
spinner and an empty body.

AppTest idioms (streamlit 1.60.0; see test_chat_expander_verify_b.py:16-33,
52-70): SessionManager is a process-global store shared with the AppTest
script thread (sid "test session id"), so state is seeded with
``SessionManager.add_message(...)`` between ``.run()`` calls. No real Ollama:
``IS_CI_TEST=true`` makes ``load_llm`` return ``GenericFakeChatModel``
(model_loader.py:455-469).
"""

import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

# Headless stub: model_loader returns fake LLM/embedders, no real Ollama calls.
# Without IS_CI_TEST the AppTest boot would load real Ollama models and hang.
os.environ.setdefault("IS_CI_TEST", "true")

sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session import SessionManager  # noqa: E402

# How long each script run may take before AppTest raises. Boot is ~1.5s.
_RUN_TIMEOUT = 60


@pytest.fixture(autouse=True)
def _reset_session_store_after():
    """공유 SessionManager 스토어('test session id') 테스트 격리.

    이 테스트가 시드한 streaming 메시지가 종료 후에도 남아 있으면, 같은
    pytest 프로세스에서 이후 실행되는 AppTest 기반 테스트(예:
    test_streamlit_app.py)가 ``chat_message[0]``에서 가이드 메시지 대신
    남은 메시지를 렌더링해 실패합니다 (order-dependent pollution).
    테스트 종료 후 ``reset()``으로 스토어를 비워 누출을 차단합니다.
    """
    yield
    SessionManager.reset()


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


def _seed_streaming(sid: str, *, status: str, thought: str, process_steps=None):
    """Seed/re-seed the streaming message.

    ``SessionManager.add_message`` merges kwargs and UPDATES in place by
    ``msg_id`` (manager.py:429-468), so re-seeding with the same
    ``msg_id`` mutates the existing message in the shared store.
    """
    SessionManager.add_message(
        "assistant",
        "준비",
        msg_type="streaming",
        status=status,
        thought=thought,
        documents=[],
        metrics={},
        msg_id="repro_1",
        session_id=sid,
        **({"process_steps": process_steps} if process_steps is not None else {}),
    )


def _answer_details(at):
    """Return the 'Answer details' expander element, or None."""
    return next((e for e in at.expander if e.label == "Answer details"), None)


def test_streaming_status_body_empty_when_thought_empty():
    """The streaming "Answer details" expander body is EMPTY when thought="";
    the sequential pipeline step still surfaces via the status/process flow."""
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)

    # (a) A seeded streaming message renders the native "Answer details"
    #     expander (the streaming renderer, chat.py:633/208).
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="")
    at.run(timeout=_RUN_TIMEOUT)

    exp = _answer_details(at)
    assert exp is not None, [e.label for e in at.expander]

    # (b) With thought="" and no process_steps, the body markdown stays EMPTY
    #     (T4/F1 "빈 박스" proof inverted: the box is present but body is empty).
    body_markdown = [m.value for m in exp.markdown]
    body_text = [t.value for t in exp.text]
    assert len(body_markdown) == 0, (
        f"expected EMPTY expander body markdown, got markdown={body_markdown}"
    )
    assert len(body_text) == 0, (
        f"expected EMPTY expander body text, got text={body_text}"
    )

    # (c) Re-seed with a different status → the SAME streaming expander persists
    #     (single fixed widget, no label churn). The status is carried on the
    #     message and consumed by the renderer for the spinner caption.
    _seed_streaming(sid, status="문서 관련성 검증 중...", thought="")
    at.run(timeout=_RUN_TIMEOUT)
    assert _answer_details(at) is not None

    # (d) Seed a thought → the body renders it: isolates thought-emptiness as
    #     the cause of the empty body.
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="생각 중...")
    at.run(timeout=_RUN_TIMEOUT)
    exp = _answer_details(at)
    thought_text = "".join(m.value for m in exp.markdown)
    assert "생각 중..." in thought_text, thought_text

    # (e) Seed process_steps → the process-steps rendering path
    #     (chat.py:280, " · ".join(steps)) fills the body even when thought="" —
    #     the sequential pipeline steps populate the expander.
    _seed_streaming(
        sid,
        status="답변 논리 설계 및 생성 중...",
        thought="",
        process_steps=["관련 지식 탐색 중...", "관련 문서 2개를 찾았습니다."],
    )
    at.run(timeout=_RUN_TIMEOUT)
    exp = _answer_details(at)
    steps_text = "".join(m.value for m in exp.markdown)
    assert "관련 지식 탐색 중..." in steps_text, steps_text
    assert not at.exception
