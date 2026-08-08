"""
Deterministic seeded AppTest repro (Q1): the streaming ``st.status`` box body
is EMPTY when ``thought=""``.

Defect documented (PRE-FIX, T1 — do NOT implement the fix here; T4 of
`.omo/plans/chat-expander-empty-body.md` rewrites scenario (b) after F1):
the streaming branch of ``_render_unified_timeline``
(`src/ui/components/chat.py:366-400`) renders the message as::

    with st.status(f"🤔 {status_text}", expanded=True, state="running") as status:
        if thought:
            status.write(thought)

so the ONLY body content is the thought (chat.py:380-381). The local
qwen3 model emits no separated reasoning, ``thought`` is always ``""``, and
the box body renders empty — while the label still advances through the
pipeline steps ("sequential title"), which is exactly the reported "빈 박스".

AppTest idioms (streamlit 1.60.0; see test_chat_expander_verify_b.py:16-33,
52-70): SessionManager is a process-global store shared with the AppTest
script thread (sid "test session id"), so state is seeded with
``SessionManager.add_message(...)`` between ``.run()`` calls. No real Ollama:
``IS_CI_TEST=true`` makes ``load_llm`` return ``GenericFakeChatModel``
(model_loader.py:455-469).

NOTE: ``st.status`` auto-transitions to "complete" at context-manager exit
(test_chat_expander_verify_b.py:100-104), so no ``state == "running"``
assertion is made here.
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


def _seed_streaming(sid: str, *, status: str, thought: str) -> None:
    """Seed/re-seed the streaming message.

    ``SessionManager.add_message`` merges kwargs and UPDATES in place by
    ``msg_id`` (manager.py:352-358), so re-seeding with the same
    ``msg_id="repro_1"`` mutates the existing message instead of duplicating
    it.
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
    )


def test_streaming_status_body_empty_when_thought_empty():
    """The streaming st.status box body is EMPTY when thought=""; the label
    still carries the sequential pipeline step (the reported "빈 박스")."""
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)

    # (a) A seeded streaming message renders a native st.status whose label is
    #     the "🤔 <step>" title. `in`, not `==`: robust to label normalization
    #     (verify_b pattern, test_chat_expander_verify_b.py:99-105).
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="")
    at.run(timeout=_RUN_TIMEOUT)

    assert at.status, f"no st.status rendered: {at.status}"
    assert "🤔 답변 논리 설계 및 생성 중..." in at.status[0].label, at.status[0].label

    # (b) POST-FIX (T4/F1) body-non-empty proof: with thought="" and no
    #     process_steps, the box body is NEVER EMPTY — the streaming branch
    #     (chat.py) renders a "준비 중..." placeholder caption. markdown/text
    #     stay empty; the placeholder is the caption.
    body_markdown = [m.value for m in at.status[0].markdown]
    body_caption = [c.value for c in at.status[0].caption]
    body_text = [t.value for t in at.status[0].text]
    assert len(body_markdown) == 0, (
        f"expected EMPTY status body markdown, got markdown={body_markdown}"
    )
    assert len(body_text) == 0, f"expected EMPTY status body text, got text={body_text}"
    assert "준비 중..." in "".join(body_caption), (
        f"expected '준비 중...' placeholder caption, got caption={body_caption}"
    )

    # (c) Re-seed with a different status → the SAME status box label changes.
    #     Proves the "sequential title" element is this st.status box, not the
    #     static final "🧠 상세 사고 과정" expander (chat.py:189).
    _seed_streaming(sid, status="문서 관련성 검증 중...", thought="")
    at.run(timeout=_RUN_TIMEOUT)

    assert "🤔 문서 관련성 검증 중..." in at.status[0].label, at.status[0].label
    assert "🤔 답변 논리 설계 및 생성 중..." not in at.status[0].label, at.status[
        0
    ].label

    # (d) Seed a thought → the body renders it: isolates thought-emptiness as
    #     the cause of the empty body (the label alone never fills the box).
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="생각 중...")
    at.run(timeout=_RUN_TIMEOUT)

    thought_text = "".join(m.value for m in at.status[0].markdown)
    assert "생각 중..." in thought_text, thought_text

    # (e) Seed process_steps → the F1 process-steps rendering path
    #     (chat.py:384-390, status.write(" · ".join(process["steps"]))) fills
    #     the body even when thought="" — the sequential pipeline steps, not
    #     just the (b) placeholder, populate the box.
    SessionManager.add_message(
        "assistant",
        "준비",
        msg_type="streaming",
        status="답변 논리 설계 및 생성 중...",
        thought="",
        documents=[],
        metrics={},
        process_steps=["관련 지식 탐색 중...", "관련 문서 2개를 찾았습니다."],
        msg_id="repro_1",
        session_id=sid,
    )
    at.run(timeout=_RUN_TIMEOUT)

    steps_text = "".join(m.value for m in at.status[0].markdown)
    assert "관련 지식 탐색 중..." in steps_text, steps_text
    assert not at.exception
