"""doc_stable_id 공용 함수 동작 검증 (R: Group 3 중복 통합)."""

from common.utils import doc_stable_id, fast_hash


class _FakeDoc:
    def __init__(self, metadata: dict | None = None, page_content: str = "") -> None:
        self.metadata = metadata or {}
        self.page_content = page_content


def test_dict_with_doc_id():
    assert doc_stable_id({"metadata": {"doc_id": "abc"}, "page_content": "x"}) == "abc"


def test_dict_doc_id_returns_str():
    assert doc_stable_id({"metadata": {"doc_id": 123}, "page_content": "x"}) == "123"


def test_dict_without_doc_id_uses_hash():
    d = {"metadata": {}, "page_content": "hello world"}
    assert doc_stable_id(d) == fast_hash("hello world")


def test_document_with_doc_id():
    assert doc_stable_id(_FakeDoc({"doc_id": "d1"}, "content")) == "d1"


def test_document_without_doc_id_uses_hash():
    doc = _FakeDoc({}, "some content")
    assert doc_stable_id(doc) == fast_hash("some content")


def test_meta_none_handled():
    """metadata가 None인 경우 빈 dict로 취급한다."""
    assert doc_stable_id(_FakeDoc(None, "c")) == fast_hash("c")


def test_missing_page_content_key():
    """page_content 키가 없는 dict는 빈 문자열로 해시한다."""
    assert doc_stable_id({"metadata": {}}) == fast_hash("")


def test_consistent_between_doc_and_dict():
    """Document와 동일 내용 dict가 같은 안정 ID를 반환한다."""
    doc = _FakeDoc({}, "same content")
    d = {"metadata": {}, "page_content": "same content"}
    assert doc_stable_id(doc) == doc_stable_id(d)


def test_doc_id_takes_priority_over_content():
    """doc_id가 있으면 page_content와 무관하게 doc_id를 사용한다."""
    d1 = {"metadata": {"doc_id": "same"}, "page_content": "aaa"}
    d2 = {"metadata": {"doc_id": "same"}, "page_content": "bbb"}
    assert doc_stable_id(d1) == doc_stable_id(d2) == "same"
