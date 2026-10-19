"""
Deterministic seeded AppTest repro: a seeded streaming message renders through
``_draw_streaming_message`` (chat.py) → ``render_generation_expander``
(chat_references.py), which draws the "Answer details" evidence expander only
when there is something to show.

Current contract (render_generation_expander):
- ``thought=""`` AND no references (documents/citations) → early return, NO
  expander is drawn at all (the strongest "no spurious content" guarantee for a
  thought-less streaming turn).
- ``thought`` only → "Thought Process" expander whose body carries the thought.
- references only → "Cited Sources" expander.
- ``thought`` + references → "Sources & Reasoning" expander.

AppTest idioms (streamlit 1.60.0; see test_chat_expander_verify_b.py): the
SessionManager store is a process-global shared with the AppTest script thread
(sid "test session id"), so state is seeded with ``SessionManager.add_message``
between ``.run()`` calls. No real Ollama: ``IS_CI_TEST=true`` makes ``load_llm``
return ``GenericFakeChatModel`` (model_loader.py).
"""

import os
import sys

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

# Headless stub: model_loader returns fake LLM/embedders, no real Ollama calls.
# Without IS_CI_TEST the AppTest boot would load real Ollama models and hang.
os.environ.setdefault("IS_CI_TEST", "true")

sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session import SessionManager  # noqa: E402

# How long each script run may take before AppTest raises. Boot is ~1.5s.
_RUN_TIMEOUT = 60

# render_generation_expander titles (strings.py, both locales) — used to locate
# the streaming evidence expander among unrelated ones (e.g. sidebar "Settings").
_GEN_EXPANDER_LABELS = (
    "Thought Process",
    "Cited Sources",
    "Sources & Reasoning",
    "사고 과정",
    "인용 출처",
    "출처 및 추론 과정",
)


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
    """Satisfy SessionManager.is_ready_for_chat (manager.py) + pin locale.

    ``ui_lang`` is pinned to "en" so the expander titles are deterministic
    (AppTest default is "en"; sidebar.py calls ``set_lang`` from this key).
    """
    SessionManager.set("ui_lang", "en", sid)
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("rag_engine", object(), sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("needs_rag_rebuild", False, sid)
    SessionManager.set("needs_qa_chain_update", False, sid)
    SessionManager.set("pdf_processing_error", None, sid)


def _seed_streaming(sid: str, *, status: str, thought: str, documents=None):
    """Seed/re-seed the streaming message.

    ``SessionManager.add_message`` merges kwargs and UPDATES in place by
    ``msg_id``, so re-seeding with the same ``msg_id`` mutates the existing
    message in the shared store.
    """
    SessionManager.add_message(
        "assistant",
        "준비",
        msg_type="streaming",
        status=status,
        thought=thought,
        documents=documents or [],
        metrics={},
        msg_id="repro_1",
        session_id=sid,
    )


def _gen_expander(at):
    """Return the streaming generation/evidence expander element, or None."""
    return next((e for e in at.expander if e.label in _GEN_EXPANDER_LABELS), None)


def _doc(page: int = 3) -> Document:
    return Document(
        page_content="참조 본문입니다.",
        metadata={"page": page, "section": "Intro"},
    )


def test_streaming_expander_reflects_thought_and_references():
    """A streaming turn's evidence expander mirrors thought/references:

    - ``thought=""`` + no refs → NO expander (empty-body guarantee);
    - ``thought`` only → "Thought Process" with the thought body;
    - refs only → "Cited Sources";
    - both → "Sources & Reasoning".
    """
    SessionManager.reset()
    at = AppTest.from_file("src/main.py").run(timeout=_RUN_TIMEOUT)
    sid = _app_session_id()
    assert not at.exception
    _set_ready_state(sid)

    # (a) thought="" and no references → the evidence expander is NOT drawn
    #     (render_generation_expander early-returns). This is the "빈 박스 없음"
    #     guarantee: a thought-less, reference-less streaming turn shows nothing
    #     spurious above the body.
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="")
    at.run(timeout=_RUN_TIMEOUT)
    assert _gen_expander(at) is None, [e.label for e in at.expander]

    # (b) thought only → "Thought Process" expander carrying the thought body.
    _seed_streaming(sid, status="답변 논리 설계 및 생성 중...", thought="생각 중...")
    at.run(timeout=_RUN_TIMEOUT)
    exp = _gen_expander(at)
    assert exp is not None, [e.label for e in at.expander]
    assert exp.label == "Thought Process", exp.label
    thought_text = "".join(m.value for m in exp.markdown)
    assert "생각 중..." in thought_text, thought_text

    # (c) references only (thought="") → "Cited Sources" expander.
    _seed_streaming(
        sid,
        status="답변 논리 설계 및 생성 중...",
        thought="",
        documents=[_doc(3)],
    )
    at.run(timeout=_RUN_TIMEOUT)
    exp = _gen_expander(at)
    assert exp is not None, [e.label for e in at.expander]
    assert exp.label == "Cited Sources", exp.label

    # (d) thought + references → "Sources & Reasoning" expander.
    _seed_streaming(
        sid,
        status="답변 논리 설계 및 생성 중...",
        thought="생각 중...",
        documents=[_doc(3)],
    )
    at.run(timeout=_RUN_TIMEOUT)
    exp = _gen_expander(at)
    assert exp is not None, [e.label for e in at.expander]
    assert exp.label == "Sources & Reasoning", exp.label
    assert "생각 중..." in "".join(m.value for m in exp.markdown)
    assert not at.exception
