from unittest.mock import patch
import pytest

from core.session import SessionManager
from ui.components import reference_plan as rp_mod
from ui.components.reference_plan import (
    _DOCS_SIG_CACHE,
    _build_reference_plan,
    _docs_signature,
    note_stream_references,
)


def _make_sample_docs():
    return [
        {
            "page_content": "Deep learning and natural language processing.",
            "metadata": {
                "doc_id": "d1",
                "page": 1,
                "source": "test.pdf",
                "file_hash": "h1",
            },
        },
        {
            "page_content": "Retrieval-augmented generation architectures.",
            "metadata": {
                "doc_id": "d2",
                "page": 2,
                "source": "test.pdf",
                "file_hash": "h1",
            },
        },
    ]


def test_docs_signature_caches_result_for_same_documents():
    """동일 문서 리스트에 대해 _docs_signature가 캐시되어 재계산하지 않는지 검증."""
    _DOCS_SIG_CACHE.clear()
    docs = _make_sample_docs()

    sig1 = _docs_signature(docs)
    assert sig1 != ""
    assert len(_DOCS_SIG_CACHE) == 1

    with patch("ui.components.reference_plan._normalize_text") as mock_norm:
        sig2 = _docs_signature(docs)
        assert sig1 == sig2
        # 캐시 적중으로 _normalize_text가 호출되지 않아야 함
        mock_norm.assert_not_called()


def test_note_stream_references_bypasses_redundant_plan_build():
    """동일한 문서로 연속 호출 시 불필요한 플랜 재계산이 조기 반환되는지 검증."""
    sid = "perf_plan_sid"
    msg_id = "perf_msg_1"
    SessionManager.reset_all_state(sid)
    SessionManager.set("pdf_file_path", "test.pdf", sid)
    docs = _make_sample_docs()

    plan1 = note_stream_references(sid, msg_id, docs, citations=[])
    assert plan1 is not None

    with patch("ui.components.reference_plan._build_reference_plan") as mock_build:
        # 인용구 없는 연속 호출 시 기존 플랜 재사용
        plan2 = note_stream_references(sid, msg_id, docs, citations=[])
        assert plan2 is plan1
        mock_build.assert_not_called()
