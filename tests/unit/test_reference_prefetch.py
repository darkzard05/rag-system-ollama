"""Task3 Wave2 — sources 이벤트 시점의 투기적 프리패치 검증.

- 불변 플랜: frozen dataclass + MappingProxy (변이 시 예외).
- discard-on-change: 최종 citations가 플랜과 다르면 폐기 후 동기 폴백.
- 순서: persist(SessionManager.add_message)가 finalize보다 먼저 수행.
- 락 경계: fitz 추출(extract_annotations_from_docs)은 세션 락 밖에서 수행.
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from core.session import SessionManager
from ui.components.reference_plan import (
    _build_reference_plan,
    adopt_prewarmed_annotations,
    discard_reference_plan,
    note_stream_references,
)


def _make_docs() -> list[dict[str, Any]]:
    """file key + page + 본문을 갖춘 결정적 fake 문서 2건."""
    return [
        {
            "page_content": "Elasticsearch dense vector search internals",
            "metadata": {
                "doc_id": "doc-a",
                "page": 3,
                "source": "C:/docs/report.pdf",
                "file_hash": "hash-aaa",
                "current_section": "Retrieval",
            },
        },
        {
            "page_content": "FlashRank reranking cross encoder scores",
            "metadata": {
                "doc_id": "doc-b",
                "page": 7,
                "source": "C:/docs/report.pdf",
                "file_hash": "hash-aaa",
                "current_section": "Reranking",
            },
        },
    ]


def _make_citations() -> list[dict[str, Any]]:
    return [
        {
            "doc_id": "doc-a",
            "page": 3,
            "section": "Retrieval",
            "text_span": "dense vector search",
        },
        {
            "doc_id": "doc-b",
            "page": 7,
            "section": "Reranking",
            "text_span": "cross encoder scores",
        },
    ]


def _seed_pdf_session(
    sid: str,
    documents: list[Any],
    citations: list[dict[str, Any]] | None = None,
    msg_id: str = "m1",
) -> None:
    SessionManager.reset_all_state(sid)
    SessionManager.set("pdf_file_path", "C:/docs/report.pdf", sid)
    SessionManager.set("file_hash", "hash-aaa", sid)
    SessionManager.set(
        "messages",
        [
            {
                "msg_id": msg_id,
                "documents": documents,
                "citations": citations or [],
                "error": None,
            }
        ],
        sid,
    )


@pytest.fixture
def pdf_sid() -> Any:
    sid = "test_ref_prefetch_pdf"
    yield sid
    SessionManager.reset_all_state(sid)


def test_plan_is_frozen_and_complete():
    """Given sources 문서/인용 / When 플랜 계산 / Then 불변이고 매핑 완비."""
    plan = _build_reference_plan(_make_docs(), _make_citations(), "m1")

    assert plan.pages == (3, 7)
    assert plan.labels[3] == "Retrieval · p.3"
    assert plan.labels[7] == "Reranking · p.7"
    assert plan.button_keys[3] == "pop_doc_m1_3_0"
    assert plan.button_keys[7] == "pop_doc_m1_7_1"
    assert plan.page_to_refs[3]["sid"] == "doc-a"
    assert "dense vector search" in plan.page_to_excerpt[3]

    with pytest.raises(FrozenInstanceError):
        plan.pages = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.page_to_refs[3] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.page_to_refs[3]["section"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.labels[3] = "mutated"  # type: ignore[index]


def test_plan_hash_stable_for_same_inputs():
    """Given 동일 입력 / When 2회 계산 / Then 동일 해시 (순서 무관)."""
    docs = _make_docs()
    cits = _make_citations()
    first = _build_reference_plan(docs, cits, "m1")
    second = _build_reference_plan(list(reversed(docs)), list(reversed(cits)), "m1")
    assert first.plan_hash == second.plan_hash
    assert first.docs_signature == second.docs_signature


def test_hit_adopts_prewarmed_annotations(pdf_sid: str):
    """Given sources 시점 prewarm / When 최종 서명 일치 / Then 논블로킹 후 콜백 채택."""
    sid = pdf_sid
    docs = _make_docs()
    cits = _make_citations()
    _seed_pdf_session(sid, docs, cits)
    sentinel = [{"page": 3, "type": "highlight"}]
    received: list[Any] = []

    def _extractor(snapshot: list[Any]) -> list[dict[str, Any]]:
        received.extend(snapshot)
        return sentinel

    plan = note_stream_references(sid, "m1", docs, cits, extractor=_extractor)
    assert plan is not None

    # 렌더 스레드는 대기하지 않고 rect 없이 반환한다.
    adopted = adopt_prewarmed_annotations(sid, "m1", docs, cits)
    assert adopted is None

    # 백그라운드 완료 시 done 콜백이 SessionManager를 하이드레이션한다.
    stored: Any = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        stored = SessionManager.get("pdf_annotations", None, sid)
        if isinstance(stored, dict) and stored.get("annotations") == sentinel:
            break
        time.sleep(0.05)
    assert isinstance(stored, dict)
    assert stored["annotations"] == sentinel
    assert stored["file_hash"] == "hash-aaa"
    # prewarm에는 세션 객체가 아닌 불변 dict 복사만 전달된다 (lock-free).
    assert len(received) == len(docs)
    for item in received:
        assert isinstance(item, dict)
    for item, orig in zip(received, docs, strict=True):
        assert item is not orig
    discard_reference_plan(sid, "m1")


def test_discard_on_citation_change(pdf_sid: str):
    """Given prewarm済 플랜 / When 최종 citations 변경 / Then 폐기(None) 후 폴백."""
    sid = pdf_sid
    docs = _make_docs()
    cits = _make_citations()
    _seed_pdf_session(sid, docs, cits)

    note_stream_references(sid, "m1", docs, cits, extractor=lambda _d: [{"page": 3}])
    changed = [dict(cits[0], text_span="completely different span")]
    changed_sig = _build_reference_plan(docs, changed, "m1").plan_hash
    assert changed_sig != _build_reference_plan(docs, cits, "m1").plan_hash

    assert adopt_prewarmed_annotations(sid, "m1", docs, changed) is None
    # 폐기되었으므로 동일 입력으로도 재채택되지 않는다.
    assert adopt_prewarmed_annotations(sid, "m1", docs, changed) is None
    discard_reference_plan(sid, "m1")


def test_miss_falls_back_to_synchronous_extract(pdf_sid: str):
    """Given 플랜/citations 불일치 / When finalize / Then 동기 추출 1회로 저장."""
    from ui.components.streaming import _finalize_pdf_side_effects

    sid = pdf_sid
    docs = _make_docs()
    cits = _make_citations()
    changed = [dict(cits[0], text_span="completely different span")]
    _seed_pdf_session(sid, docs, changed)
    note_stream_references(sid, "m1", docs, cits, extractor=lambda _d: [{"page": 3}])
    fallback = [{"page": 3, "type": "sync"}]

    with patch(
        "ui.components.streaming.extract_annotations_from_docs",
        return_value=fallback,
    ) as mock_extract:
        _finalize_pdf_side_effects(sid, "m1")

    mock_extract.assert_called_once()
    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored is not None
    assert stored["annotations"] == fallback
    assert stored["file_hash"] == "hash-aaa"
    discard_reference_plan(sid, "m1")


def test_no_lock_held_during_fitz(pdf_sid: str):
    """Given 완료 턴 / When finalize / Then fitz 호출 시점에 세션 락 미보유."""
    from ui.components.streaming import _finalize_pdf_side_effects

    sid = pdf_sid
    _seed_pdf_session(sid, _make_docs(), _make_citations())

    class _RecordingLock:
        def __init__(self) -> None:
            self.held = False

        def __enter__(self) -> _RecordingLock:
            self.held = True
            return self

        def __exit__(self, *exc: Any) -> bool:
            self.held = False
            return False

    recorder = _RecordingLock()

    def _assert_unlocked(documents: list[Any]) -> list[dict[str, Any]]:
        assert not recorder.held, "fitz 추출은 세션 락 밖에서 수행되어야 한다"
        return [{"page": 1}]

    with (
        patch.object(SessionManager, "_acquire_lock", return_value=recorder),
        patch(
            "ui.components.streaming.extract_annotations_from_docs",
            side_effect=_assert_unlocked,
        ),
    ):
        _finalize_pdf_side_effects(sid, "m1")

    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored is not None
    assert stored["annotations"] == [{"page": 1}]


def test_persist_precedes_finalize(monkeypatch: Any) -> None:
    """Given 스트림 소비 / When 완료 / Then persist가 finalize보다 먼저 수행."""
    import ui.components.streaming as streaming_mod

    sid = "test_ref_prefetch_order"
    SessionManager.reset_all_state(sid)
    order: list[str] = []

    orig_add = SessionManager.__dict__["add_message"]

    def _spy_add(cls: Any, *args: Any, **kwargs: Any) -> Any:
        order.append("persist")
        return orig_add.__func__(cls, *args, **kwargs)

    def _spy_finalize(fin_sid: str, msg_id: str) -> None:
        order.append("finalize")

    monkeypatch.setattr(SessionManager, "add_message", classmethod(_spy_add))
    monkeypatch.setattr(streaming_mod, "_finalize_pdf_side_effects", _spy_finalize)

    def _fake_stream(query: str, model_name: str, session_id: str) -> Any:
        yield SimpleNamespace(
            status=None,
            thought=None,
            content="답변",
            metadata={"documents": _make_docs()},
            performance=None,
        )

    monkeypatch.setattr(streaming_mod, "stream_chunks", _fake_stream)

    result = streaming_mod.consume_stream_into_message(sid, "질문", "test-model")

    assert result is not None
    assert order[-1] == "finalize"
    last_persist = max(i for i, item in enumerate(order) if item == "persist")
    assert last_persist < len(order) - 1
    SessionManager.reset_all_state(sid)
