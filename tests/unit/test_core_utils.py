from langchain_core.documents import Document

from common.utils import apply_tooltips_to_response, normalize_latex_delimiters
from core.graph_builder import _merge_adjacent_chunks

# --- Test _merge_adjacent_chunks ---


def test_merge_adjacent_chunks_empty():
    assert _merge_adjacent_chunks([]) == []


def test_merge_adjacent_chunks_single():
    doc = Document(
        page_content="Content",
        metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
    )
    assert _merge_adjacent_chunks([doc])[0].page_content == "Content"


def test_merge_adjacent_chunks_no_consecutive():
    docs = [
        Document(
            page_content="Content 1",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="Content 2",
            metadata={"source": "a.pdf", "page": 3, "chunk_index": 0},
        ),
        Document(
            page_content="Content 3",
            metadata={"source": "b.pdf", "page": 1, "chunk_index": 0},
        ),
    ]
    result = _merge_adjacent_chunks(docs)
    assert len(result) == 3
    # 정렬 순서 확인 (source a -> b, page 1 -> 3)
    assert result[0].page_content == "Content 1"
    assert result[1].page_content == "Content 2"
    assert result[2].page_content == "Content 3"


def test_merge_adjacent_chunks_simple_consecutive():
    docs = [
        Document(
            page_content="Part 1.",
            metadata={
                "source": "a.pdf",
                "page": 1,
                "chunk_index": 0,
                "current_section": "Sec1",
                "start_index": 0,
                "end_index": 10,
            },
        ),
        Document(
            page_content="Part 2.",
            metadata={
                "source": "a.pdf",
                "page": 1,
                "chunk_index": 1,
                "current_section": "Sec1",
                "start_index": 15,
                "end_index": 25,
            },
        ),
        Document(
            page_content="Part 3.",
            metadata={
                "source": "a.pdf",
                "page": 1,
                "chunk_index": 2,
                "current_section": "Sec1",
                "start_index": 30,
                "end_index": 40,
            },
        ),
    ]
    result = _merge_adjacent_chunks(docs)
    assert len(result) == 1
    # \n\n 구분자 확인
    assert result[0].page_content == "Part 1.\n\nPart 2.\n\nPart 3."
    assert result[0].metadata["end_index"] == 40


# --- Test apply_tooltips_to_response (src/common/utils.py) ---
def test_apply_tooltips_to_response_empty_inputs():
    assert apply_tooltips_to_response("", []) == ""
    assert apply_tooltips_to_response("text", []) == "text"
    doc = Document(page_content="Test", metadata={"page": 1})
    assert apply_tooltips_to_response("", [doc]) == ""


def test_apply_tooltips_to_response_no_page_metadata():
    doc = Document(page_content="Test content", metadata={})
    # [수정] 인용구 패턴 [1]에서 페이지 1을 추출하므로, 문서에 해당 페이지가 없더라도 
    # 기본 메시지와 함께 툴팁 태그로 감싸집니다.
    result = apply_tooltips_to_response("[1]", [doc])
    assert "citation-highlight" in result
    assert "인용된 원문 정보를 불러올 수 없습니다." in result


def test_apply_tooltips_to_response_basic():
    doc = Document(page_content="This is a test sentence.", metadata={"page": 5})
    response = "See [p.5]."
    result = apply_tooltips_to_response(response, [doc])

    assert "citation-highlight" in result
    assert 'data-page="5"' in result
    assert "This is a test sentence." in result


# --- Test normalize_latex_delimiters ---
def test_normalize_latex_delimiters():
    # \( \) -> $ $
    assert normalize_latex_delimiters(r"Text \(x^2\) text") == "Text $x^2$ text"
    # \[ \] -> $$ $$
    assert normalize_latex_delimiters(r"\[y = ax + b\]") == "$$y = ax + b$$"
    # 중복 방지
    assert normalize_latex_delimiters("$x$") == "$x$"
