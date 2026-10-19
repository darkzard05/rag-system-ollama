from unittest.mock import MagicMock, patch

import pytest

import ui.components.chat_references as refs
from ui.components.chat_references import (
    clear_reference_rows_cache,
    render_reference_list,
)


def _doc(page: int, text: str, doc_id: str | None = None) -> MagicMock:
    doc = MagicMock()
    meta = {"page": page}
    if doc_id is not None:
        meta["doc_id"] = doc_id
    doc.metadata = meta
    doc.page_content = text
    return doc


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_reference_rows_cache()
    yield
    clear_reference_rows_cache()


def test_reference_list_caches_rows_and_skips_normalize_excerpt():
    """동일 메시지 재렌더 시 normalize_excerpt를 재호출하지 않고 캐시된 rows를 사용하는지 검증."""
    docs = [_doc(1, "테스트 본문 내용입니다.", doc_id="d1")]
    st_mock = MagicMock()

    with (
        patch.object(refs, "st", st_mock),
        patch.object(
            refs, "normalize_excerpt", wraps=refs.normalize_excerpt
        ) as spy_norm,
    ):
        # 1회차 렌더링 (캐시 미스 -> 계산 및 캐싱)
        ok1 = render_reference_list("msg_cached", docs, citations=None)
        assert ok1 is True
        assert spy_norm.call_count == 1

        # 2회차 렌더링 (동일 msg_id + 문서 -> 캐시 히트)
        ok2 = render_reference_list("msg_cached", docs, citations=None)
        assert ok2 is True
        # 캐시 히트로 인해 normalize_excerpt 추가 호출이 0회여야 함
        assert spy_norm.call_count == 1

        # 버튼은 두 번 모두 정상적으로 렌더링되어야 함
        assert st_mock.button.call_count == 2
