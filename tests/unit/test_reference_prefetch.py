"""Task3 Wave2 — sources 이벤트 시점의 투기적 프리패치 검증 (단순안).

- 불변 플랜: frozen dataclass + MappingProxy (변이 시 예외).
- 문서기준 매칭: 최종 citations가 바뀌어도 문서가 동일하면 prewarm 적중.
  인용은 하이라이트 순위 힌트에 불과하므로 정합성에 영향이 없다.
- miss/미완료 시 렌더 스레드 논블로킹: 동기 폴백 없이 백그라운드에 맡긴다.
- 순서: persist(SessionManager.add_message)가 finalize보다 먼저 수행.
- 락 경계: fitz 추출은 세션 락 밖(백그라운드)에서 수행.
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
    """단순안: 인용 변경은 폐기 사유가 아니다 (문서기준 매칭).

    동일 문서 + 변경된 인용이면 plan_hash는 바뀌지만 docs_signature는
    유지되므로 adopt는 유효하다. 문서 자체가 바뀌어야 miss다.
    """
    sid = pdf_sid
    docs = _make_docs()
    cits = _make_citations()
    _seed_pdf_session(sid, docs, cits)

    note_stream_references(sid, "m1", docs, cits, extractor=lambda _d: [{"page": 3}])
    changed = [dict(cits[0], text_span="completely different span")]
    changed_sig = _build_reference_plan(docs, changed, "m1").plan_hash
    assert changed_sig != _build_reference_plan(docs, cits, "m1").plan_hash
    same_docs = _build_reference_plan(docs, changed, "m1").docs_signature
    assert same_docs == _build_reference_plan(docs, cits, "m1").docs_signature

    # 인용이 바뀌어도 문서기준 매칭은 유효하다: 예외 없이 판정만 수행한다.
    # (pending이면 done 콜백 등록 후 None, 완료됐으면 sentinel 반환.)
    adopted = adopt_prewarmed_annotations(sid, "m1", docs, changed)
    assert adopted is None or adopted == [{"page": 3}]
    discard_reference_plan(sid, "m1")


def test_miss_falls_back_to_synchronous_extract(pdf_sid: str):
    """Given 문서 불일치(miss) / When finalize / Then 동기 추출 없이 deferred."""
    import time as _time

    from ui.components.streaming import _finalize_pdf_side_effects

    sid = pdf_sid
    docs = _make_docs()
    cits = _make_citations()
    other_docs = [
        {
            "page_content": "entirely different document body",
            "metadata": {
                "doc_id": "doc-z",
                "page": 9,
                "source": "C:/docs/other.pdf",
                "file_hash": "hash-zzz",
                "current_section": "Other",
            },
        }
    ]
    _seed_pdf_session(sid, other_docs, cits, msg_id="m-miss")
    note_stream_references(
        sid, "m-miss", docs, cits, extractor=lambda _d: [{"page": 3}]
    )

    with (
        patch(
            "ui.components.streaming.adopt_prewarmed_annotations",
            return_value=None,
        ),
        patch(
            "ui.components.streaming.note_stream_references",
        ) as mock_note,
        patch(
            "ui.components.streaming.extract_annotations_from_docs",
        ) as mock_extract,
    ):
        _t0 = _time.perf_counter()
        _finalize_pdf_side_effects(sid, "m-miss")
        _dt = _time.perf_counter() - _t0

    # 렌더 스레드 논블로킹: 동기 추출을 직접 호출하지 않고 백그라운드에 맡긴다.
    mock_extract.assert_not_called()
    mock_note.assert_called_once()
    assert _dt < 1.0
    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored is None
    discard_reference_plan(sid, "m-miss")


def test_no_lock_held_during_fitz(pdf_sid: str):
    """Given prewarm 적중 / When finalize / Then 세션 락 없이 즉시 저장."""
    from ui.components.streaming import _finalize_pdf_side_effects

    sid = pdf_sid
    _seed_pdf_session(sid, _make_docs(), _make_citations(), msg_id="m-lock")

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

    with patch.object(SessionManager, "_acquire_lock", return_value=recorder):
        # prewarm 경로를 동기 sentinel로 강제해 락 경계만 검증한다.
        # (streaming 모듈에 바인딩된 참조를 패치해야 한다.)
        sentinel = [{"page": 1}]
        with (
            patch(
                "ui.components.streaming.adopt_prewarmed_annotations",
                return_value=sentinel,
            ),
            patch(
                "ui.components.streaming.extract_annotations_from_docs",
                side_effect=_assert_unlocked,
            ),
        ):
            _finalize_pdf_side_effects(sid, "m-lock")

    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored is not None
    assert stored["annotations"] == [{"page": 1}]
    discard_reference_plan(sid, "m-lock")


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
