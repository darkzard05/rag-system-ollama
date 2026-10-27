import sys

from langchain_core.documents import Document

from common.text_utils import split_by_headers
from core.model_loader import is_ollama_embedder
from core.retriever_factory import to_lean_documents
from ui.components.common import format_chat_display_text


def test_to_lean_documents_contract():
    """Document 객체 및 dict 형태의 메타데이터에서 좌표 키가 완벽히 제거되는지 검증"""
    doc_obj = Document(
        page_content="test content 1",
        metadata={
            "page": 1,
            "word_coords": [[0, 0, 10, 10, "word"]],
            "file_hash": "abc",
        },
    )
    doc_dict = {
        "page_content": "test content 2",
        "metadata": {"page": 2, "page_coords": {1: [0, 0, 10, 10]}, "score": 0.9},
    }

    lean = to_lean_documents([doc_obj, doc_dict, "raw_string"])

    # Assertions
    assert isinstance(lean[0], Document)
    assert "word_coords" not in lean[0].metadata
    assert lean[0].metadata["file_hash"] == "abc"

    assert isinstance(lean[1], dict)
    assert "page_coords" not in lean[1]["metadata"]
    assert lean[1]["metadata"]["score"] == 0.9

    assert lean[2] == "raw_string"


def test_split_by_headers_contract():
    """단일화된 split_by_headers가 헤더와 본문을 정확하게 분리하는지 검증"""
    md_text = "# Header 1\nBody 1\n## Header 2\nBody 2"
    sections = split_by_headers(md_text)
    assert len(sections) == 2
    assert sections[0] == ("Header 1", "\nBody 1\n")
    assert sections[1] == ("Header 2", "\nBody 2")

    no_header_text = "Plain body text without header"
    assert split_by_headers(no_header_text) == [("", no_header_text)]


def test_format_chat_display_text_contract():
    """어시스턴트 및 사용자 메시지 정규화 파이프라인의 통합 일관성 검증"""
    raw_assistant = "<script>alert('xss')</script> \\[ x^2 \\] [doc:abc1234] [score:0.95] Final Answer."
    formatted = format_chat_display_text(raw_assistant, is_assistant=True)

    # 1. HTML escape 적용 확인
    assert "&lt;script&gt;" in formatted
    # 2. LaTeX 블록 구분자 정규화 확인 ($$ x^2 $$)
    assert "$$ x^2 $$" in formatted
    # 3. Context 메타 토큰 제거 확인
    assert "[doc:abc1234]" not in formatted
    assert "[score:0.95]" not in formatted
    assert "Final Answer." in formatted

    raw_user = "<User> message with [doc:keep] preserved"
    formatted_user = format_chat_display_text(raw_user, is_assistant=False)
    assert formatted_user == "<User> message with [doc:keep] preserved"


def test_is_ollama_embedder_nested_unwrap(monkeypatch):
    """MemoizingEmbedding 등 다층 래퍼 구조에서 원본 OllamaEmbeddings 인스턴스를 정확히 식별하는지 검증"""

    class DummyInner:
        pass

    class MockOllama:
        pass

    # Mock langchain_ollama in sys.modules for test (auto-restored by monkeypatch).
    from types import ModuleType

    m = ModuleType("langchain_ollama")
    m.OllamaEmbeddings = MockOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", m)

    class Wrapper:
        def __init__(self, inner):
            self._inner = inner

    nested = Wrapper(Wrapper(MockOllama()))
    assert is_ollama_embedder(nested) is True

    non_ollama = Wrapper(Wrapper(DummyInner()))
    assert is_ollama_embedder(non_ollama) is False
