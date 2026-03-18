from langchain_core.documents import Document

from common.utils import normalize_latex_delimiters
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


# --- Test normalize_latex_delimiters ---
def test_normalize_latex_delimiters():
    # \( \) -> $ $
    assert normalize_latex_delimiters(r"Text \(x^2\) text") == "Text $x^2$ text"
    # \[ \] -> $$ $$
    assert normalize_latex_delimiters(r"\[y = ax + b\]") == "$$y = ax + b$$"
    # 중복 방지
    assert normalize_latex_delimiters("$x$") == "$x$"
