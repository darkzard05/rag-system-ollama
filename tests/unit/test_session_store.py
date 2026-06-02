# SessionStore의 데이터 저장 및 관리를 검증하는 단위 테스트.
import pytest
from core.session.store import SessionStore


def test_store_set_get():
    store = SessionStore()
    store.set("key1", "value1", session_id="s1")
    assert store.get("key1", session_id="s1") == "value1"
    assert store.get("key1", session_id="s2") is None


def test_store_clear():
    store = SessionStore()
    store.set("key1", "value1", session_id="s1")
    store.clear("s1")
    assert store.get("key1", session_id="s1") is None
