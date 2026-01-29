import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from langchain_core.documents import Document

# Optimized functions
from src.core.graph_builder import _merge_consecutive_chunks
from src.common.utils import (
    apply_tooltips_to_response,
    preprocess_text,
    clean_query_text,
)
from src.core.rag_core import _load_pdf_docs  # Need to mock fitz for this
from src.core.semantic_chunker import (
    EmbeddingBasedSemanticChunker,
)  # Need to mock embedder for this


# Mock for Semantic Chunker
class MockEmbedder:
    def embed_documents(self, texts):
        # Return dummy embeddings for testing
        return np.array(
            [
                [float(ord(c)) for c in text[:10]] + [0] * (10 - len(text))
                for text in texts
            ]
        )


# --- Test _merge_consecutive_chunks (src/core/graph_builder.py) ---
def test_merge_consecutive_chunks_empty():
    assert _merge_consecutive_chunks([]) == []


def test_merge_consecutive_chunks_single_doc():
    doc = Document(
        page_content="Content",
        metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
    )
    assert _merge_consecutive_chunks([doc]) == [doc]


def test_merge_consecutive_chunks_no_consecutive():
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
    assert _merge_consecutive_chunks(docs) == docs


def test_merge_consecutive_chunks_simple_consecutive():
    docs = [
        Document(
            page_content="Part 1.",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="Part 2.",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="Part 3.",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 2},
        ),
    ]
    merged_doc = Document(
        page_content="Part 1. Part 2. Part 3.",
        metadata={"source": "a.pdf", "page": 1, "chunk_index": 2},
    )
    assert _merge_consecutive_chunks(docs) == [merged_doc]


def test_merge_consecutive_chunks_mixed_consecutive():
    docs = [
        Document(
            page_content="P1-C1",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="P1-C2",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="P2-C1",
            metadata={"source": "a.pdf", "page": 2, "chunk_index": 0},
        ),
        Document(
            page_content="P2-C2",
            metadata={"source": "a.pdf", "page": 2, "chunk_index": 1},
        ),
        Document(
            page_content="B-P1-C1",
            metadata={"source": "b.pdf", "page": 1, "chunk_index": 0},
        ),
    ]
    expected = [
        Document(
            page_content="P1-C1 P1-C2",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="P2-C1 P2-C2",
            metadata={"source": "a.pdf", "page": 2, "chunk_index": 1},
        ),
        Document(
            page_content="B-P1-C1",
            metadata={"source": "b.pdf", "page": 1, "chunk_index": 0},
        ),
    ]
    assert _merge_consecutive_chunks(docs) == expected


def test_merge_consecutive_chunks_missing_metadata():
    docs = [
        Document(page_content="Content 1", metadata={"source": "a.pdf"}),
        Document(page_content="Content 2", metadata={"source": "a.pdf", "page": 1}),
        Document(
            page_content="Content 3",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="Content 4",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 1},
        ),
    ]
    expected = [
        Document(page_content="Content 1", metadata={"source": "a.pdf"}),
        Document(page_content="Content 2", metadata={"source": "a.pdf", "page": 1}),
        Document(
            page_content="Content 3 Content 4",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 1},
        ),
    ]
    assert _merge_consecutive_chunks(docs) == expected


def test_merge_consecutive_chunks_large_number_of_docs(benchmark):
    base_doc = Document(
        page_content="test content",
        metadata={"source": "large.pdf", "page": 1, "chunk_index": 0},
    )
    docs = [
        Document(
            page_content=f"content {i}",
            metadata={
                "source": "large.pdf",
                "page": i // 100 + 1,
                "chunk_index": i % 100,
            },
        )
        for i in range(1000)
    ]

    # Simulate partial consecutiveness
    for i in range(0, 1000, 2):
        docs[i].metadata["page"] = i // 2 + 1  # break consecutiveness for some
        docs[i].metadata["chunk_index"] = 0
        if i + 1 < 1000:
            docs[i + 1].metadata["page"] = i // 2 + 1
            docs[i + 1].metadata["chunk_index"] = 1

    benchmark(_merge_consecutive_chunks, docs)


# --- Test apply_tooltips_to_response (src/common/utils.py) ---
def test_apply_tooltips_to_response_empty_inputs():
    assert apply_tooltips_to_response("", []) == ""
    assert apply_tooltips_to_response("text", []) == "text"
    doc = Document(page_content="Test", metadata={"page": 1})
    assert apply_tooltips_to_response("", [doc]) == ""


def test_apply_tooltips_to_response_no_page_metadata():
    docs = [Document(page_content="Content", metadata={"source": "a.pdf"})]
    response = "This is a response [p.1]."
    assert apply_tooltips_to_response(response, docs) == response


def test_apply_tooltips_to_response_no_citations_in_response():
    docs = [Document(page_content="Page 1 Content.", metadata={"page": 1})]
    response = "Just plain text."
    assert apply_tooltips_to_response(response, docs) == response


def test_apply_tooltips_to_response_valid_single_citation():
    docs = [Document(page_content="Content from page 1.", metadata={"page": 1})]
    response = "This is a response [p.1]."
    expected_text = 'This is a response <span class="tooltip">[p.1]<span class="tooltip-text">Content from page 1.</span></span>.'
    assert apply_tooltips_to_response(response, docs) == expected_text


def test_apply_tooltips_to_response_valid_multiple_citations_same_page():
    docs = [
        Document(page_content="Part A from page 2.", metadata={"page": 2}),
        Document(page_content="Part B from page 2.", metadata={"page": 2}),
    ]
    response = "Info [p.2] and more info [P.2]."
    expected_tooltip = "Part A from page 2.\n\n... Part B from page 2."
    expected_text = f'Info <span class="tooltip">[p.2]<span class="tooltip-text">{expected_tooltip}</span></span> and more info <span class="tooltip">[P.2]<span class="tooltip-text">{expected_tooltip}</span></span>.'
    assert apply_tooltips_to_response(response, docs) == expected_text


def test_apply_tooltips_to_response_citation_for_non_existent_page():
    docs = [Document(page_content="Content from page 1.", metadata={"page": 1})]
    response = "This is a response [p.99]."
    assert (
        apply_tooltips_to_response(response, docs) == response
    )  # No tooltip for non-existent page


def test_apply_tooltips_to_response_long_page_content_truncation():
    long_content = "A" * 600 + "B" * 100
    docs = [Document(page_content=long_content, metadata={"page": 1})]
    response = "Long content [p.1]."
    expected_tooltip_content = "A" * 500 + "..."
    expected_text = f'Long content <span class="tooltip">[p.1]<span class="tooltip-text">{expected_tooltip_content}</span></span>.'
    assert apply_tooltips_to_response(response, docs) == expected_text


def test_apply_tooltips_to_response_html_special_chars_escaped():
    html_content = "Content with <tag> & \"quotes\" 'single'."
    docs = [Document(page_content=html_content, metadata={"page": 1})]
    response = "Special chars [p.1]."
    escaped_content = "Content with &lt;tag&gt; &amp; &quot;quotes&quot; &#x27;single'."
    expected_text = f'Special chars <span class="tooltip">[p.1]<span class="tooltip-text">{escaped_content}</span></span>.'
    assert apply_tooltips_to_response(response, docs) == expected_text


# --- Test preprocess_text (src/common/utils.py) ---
def test_preprocess_text_empty():
    assert preprocess_text("") == ""


def test_preprocess_text_null_chars():
    assert preprocess_text("Hello\x00World") == "Hello World"


def test_preprocess_text_multiple_spaces():
    assert preprocess_text("  Hello   World  ") == "Hello World"


def test_preprocess_text_newlines():
    assert preprocess_text("Hello\nWorld\r\nTest") == "Hello World Test"


def test_preprocess_text_normal_string():
    assert preprocess_text("This is a test string.") == "This is a test string."


# --- Test clean_query_text (src/common/utils.py) ---
def test_clean_query_text_empty():
    assert clean_query_text("") == ""


def test_clean_query_text_leading_trailing_spaces():
    assert clean_query_text("  query  ") == "query"


def test_clean_query_text_with_prefixes():
    assert clean_query_text("1. This is a query.") == "This is a query."
    assert clean_query_text("- A query.") == "A query."
    assert clean_query_text("Example: Another query.") == "Another query."
    assert clean_query_text("Query: What is it?") == "What is it?"
    assert clean_query_text("Question: Explain.") == "Explain."
    assert clean_query_text(" 1) What is it?") == "What is it?"


def test_clean_query_text_with_quotes():
    assert clean_query_text('"This is a query."') == "This is a query."
    assert clean_query_text("'Another query.'") == "Another query."
    assert clean_query_text(' "Mixed quotes" ') == "Mixed quotes"


def test_clean_query_text_combination():
    assert clean_query_text('1. "A combined query."') == "A combined query."
    assert clean_query_text(" Question: 'Final query.' ") == "Final query."


def test_clean_query_text_special_chars_preserved():
    assert clean_query_text("What about C++ and .NET?") == "What about C++ and .NET?"
    assert clean_query_text("Is Python 3.9 supported?") == "Is Python 3.9 supported?"


# --- Test _load_pdf_docs (src/core/rag_core.py) ---
@patch("src.core.rag_core.fitz.open")
@patch("src.core.rag_core.SessionManager")
@patch("src.core.rag_core.monitor")
def test_load_pdf_docs_empty_pdf(mock_monitor, mock_session_manager, mock_fitz_open):
    mock_doc_file = MagicMock()
    mock_doc_file.__len__.return_value = 0
    mock_doc_file.__enter__.return_value = mock_doc_file
    mock_doc_file.__exit__.return_value = False
    mock_fitz_open.return_value = mock_doc_file

    docs = _load_pdf_docs("dummy.pdf", "dummy.pdf")
    assert docs == []
    mock_session_manager.add_status_log.assert_called()


@patch("src.core.rag_core.fitz.open")
@patch("src.core.rag_core.SessionManager")
@patch("src.core.rag_core.monitor")
def test_load_pdf_docs_single_page_valid_text(
    mock_monitor, mock_session_manager, mock_fitz_open
):
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 content."

    mock_doc_file = MagicMock()
    mock_doc_file.__len__.return_value = 1
    mock_doc_file.__enter__.return_value = mock_doc_file
    mock_doc_file.__iter__.return_value = [mock_page]
    mock_doc_file.__exit__.return_value = False
    mock_fitz_open.return_value = mock_doc_file

    docs = _load_pdf_docs("valid.pdf", "valid.pdf")
    assert len(docs) == 1
    assert docs[0].page_content == "Page 1 content."
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "valid.pdf"
    assert docs[0].metadata["total_pages"] == 1


@patch("src.core.rag_core.fitz.open")
@patch("src.core.rag_core.SessionManager")
@patch("src.core.rag_core.monitor")
def test_load_pdf_docs_multiple_pages_mixed_content(
    mock_monitor, mock_session_manager, mock_fitz_open
):
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Content for page one."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Short"  # Will be filtered out
    mock_page3 = MagicMock()
    mock_page3.get_text.return_value = (
        "Content for page three, which is longer than ten chars."
    )

    mock_doc_file = MagicMock()
    mock_doc_file.__len__.return_value = 3
    mock_doc_file.__enter__.return_value = mock_doc_file
    mock_doc_file.__iter__.return_value = [mock_page1, mock_page2, mock_page3]
    mock_doc_file.__exit__.return_value = False
    mock_fitz_open.return_value = mock_doc_file

    docs = _load_pdf_docs("mixed.pdf", "mixed.pdf")
    assert len(docs) == 2
    assert docs[0].page_content == "Content for page one."
    assert docs[0].metadata["page"] == 1
    assert (
        docs[1].page_content
        == "Content for page three, which is longer than ten chars."
    )
    assert docs[1].metadata["page"] == 3


@patch("src.core.rag_core.fitz.open")
@patch("src.core.rag_core.SessionManager")
@patch("src.core.rag_core.monitor")
def test_load_pdf_docs_get_text_exception(
    mock_monitor, mock_session_manager, mock_fitz_open
):
    mock_page = MagicMock()
    mock_page.get_text.side_effect = Exception("PDF parsing error")

    mock_doc_file = MagicMock()
    mock_doc_file.__len__.return_value = 1
    mock_doc_file.__enter__.return_value = mock_doc_file
    mock_doc_file.__iter__.return_value = [mock_page]
    mock_doc_file.__exit__.return_value = False
    mock_fitz_open.return_value = mock_doc_file

    docs = _load_pdf_docs("error.pdf", "error.pdf")
    assert docs == []
    mock_session_manager.replace_last_status_log.assert_called()  # Should log warning


# --- Test EmbeddingBasedSemanticChunker (src/core/semantic_chunker.py) ---
@pytest.fixture
def semantic_chunker_instance():
    # Setup a chunker instance with mocked embedder
    return EmbeddingBasedSemanticChunker(
        embedder=MockEmbedder(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_value=75.0,
        min_chunk_size=10,
        max_chunk_size=50,
        similarity_threshold=0.5,
        batch_size=2,
    )


def test_find_breakpoints_empty_similarities(semantic_chunker_instance):
    assert semantic_chunker_instance._find_breakpoints([]) == []


def test_find_breakpoints_no_breakpoints(semantic_chunker_instance):
    # All similarities above threshold (e.g., threshold at 75th percentile of [0.8, 0.9, 0.85] would be ~0.88)
    # If values are high, threshold is high. If breakpoint_threshold_value is 75.0, it means 75th percentile.
    # so threshold is a low similarity value.
    semantic_chunker_instance.breakpoint_threshold_type = "percentile"
    semantic_chunker_instance.breakpoint_threshold_value = (
        25.0  # Low percentile means threshold is low, so more breakpoints
    )

    similarities = [0.8, 0.9, 0.85, 0.7, 0.95]
    # For 25th percentile, threshold might be around 0.8 (value below which 25% of data falls)
    # If threshold is e.g., 0.8:
    # 0.8 (index 0) is not < 0.8
    # 0.7 (index 3) is < 0.8
    # If the threshold is for example 0.75, only 0.7 is lower
    # A more robust test:
    similarities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 25th percentile of this list is 0.3. threshold will be 0.3
    # Breakpoints where sim < 0.3 -> indices 0 (0.1), 1 (0.2)
    breakpoints = semantic_chunker_instance._find_breakpoints(similarities)
    assert breakpoints == [1, 2]  # breakpoints are 1-indexed positions


def test_find_breakpoints_multiple_breakpoints(semantic_chunker_instance):
    semantic_chunker_instance.breakpoint_threshold_type = "similarity_threshold"
    semantic_chunker_instance.breakpoint_threshold_value = 0.5
    similarities = [0.8, 0.2, 0.7, 0.3, 0.9]
    assert semantic_chunker_instance._find_breakpoints(similarities) == [2, 4]


def test_optimize_chunk_sizes_empty(semantic_chunker_instance):
    assert semantic_chunker_instance._optimize_chunk_sizes([]) == []


def test_optimize_chunk_sizes_single_chunk(semantic_chunker_instance):
    chunk = {"text": "Single chunk content.", "start": 0, "end": 20}
    assert semantic_chunker_instance._optimize_chunk_sizes([chunk]) == [chunk]


def test_optimize_chunk_sizes_merge_within_max_size(semantic_chunker_instance):
    # min_chunk_size=10, max_chunk_size=50
    chunks = [
        {"text": "Hello world.", "start": 0, "end": 12},  # len 12
        {"text": "How are you?", "start": 13, "end": 25},  # len 12
        {"text": "Fine thanks.", "start": 26, "end": 38},  # len 12
    ]
    # Merged len: 12 + 1 + 12 = 25 (ok)
    # Merged len: 25 + 1 + 12 = 38 (ok)
    expected = [
        {"text": "Hello world. How are you? Fine thanks.", "start": 0, "end": 38}
    ]
    assert semantic_chunker_instance._optimize_chunk_sizes(chunks) == expected


def test_optimize_chunk_sizes_split_exceeds_max_size(semantic_chunker_instance):
    # min_chunk_size=10, max_chunk_size=50
    chunks = [
        {"text": "A" * 30, "start": 0, "end": 30},
        {"text": "B" * 30, "start": 31, "end": 61},
    ]
    # Merged len: 30 + 1 + 30 = 61 (exceeds 50)
    expected = [
        {"text": "A" * 30, "start": 0, "end": 30},
        {"text": "B" * 30, "start": 31, "end": 61},
    ]
    assert semantic_chunker_instance._optimize_chunk_sizes(chunks) == expected


def test_optimize_chunk_sizes_large_chunk_remains_unmerged(semantic_chunker_instance):
    # min_chunk_size=10, max_chunk_size=50
    chunk = {"text": "C" * 100, "start": 0, "end": 100}  # Already too large
    assert semantic_chunker_instance._optimize_chunk_sizes([chunk]) == [chunk]


def test_split_text_empty_input(semantic_chunker_instance):
    assert semantic_chunker_instance.split_text("") == []
    assert semantic_chunker_instance.split_text("   ") == []


def test_split_text_single_sentence(semantic_chunker_instance):
    # Mock _split_sentences to return a single sentence
    with patch.object(
        semantic_chunker_instance,
        "_split_sentences",
        return_value=[{"text": "Hello world.", "start": 0, "end": 12}],
    ):
        result = semantic_chunker_instance.split_text("Hello world.")
        assert len(result) == 1
        assert result[0]["text"] == "Hello world."


def test_split_text_consecutive_short_sentences_merge(semantic_chunker_instance):
    # min_chunk_size=10, max_chunk_size=50, len < 3 will merge if current_s < min_chunk_size/2
    with patch.object(
        semantic_chunker_instance,
        "_split_sentences",
        return_value=[
            {"text": "Hi.", "start": 0, "end": 3},  # len 3
            {"text": "How are you?", "start": 4, "end": 16},  # len 12
            {"text": "I am fine.", "start": 17, "end": 27},  # len 10
        ],
    ):
        result = semantic_chunker_instance.split_text("Hi. How are you? I am fine.")
        # Given the min_chunk_size of 10, "Hi" might merge with "How are you?"
        # The condition is len(s["text"].strip()) < 3 OR (len(merged_so_far) < min_chunk_size/2 AND len(s["text"]) < min_chunk_size)
        # "Hi." (3 chars) is not < 3. But it is < min_chunk_size/2 (5)
        # So "Hi. How are you?" should merge.
        # "Hi. How are you?" (16 chars)
        # "I am fine." (10 chars)
        # So it should be two chunks.
        assert len(result) == 2
        assert result[0]["text"] == "Hi. How are you?"
        assert result[1]["text"] == "I am fine."


def test_split_text_short_sentences_no_merge_due_to_min_chunk_size(
    semantic_chunker_instance,
):
    # If the current_s_parts is already long, new short sentences won't merge
    with patch.object(
        semantic_chunker_instance,
        "_split_sentences",
        return_value=[
            {
                "text": "This is a long sentence that already exceeds half min_chunk_size.",
                "start": 0,
                "end": 66,
            },  # len 66 > 10/2=5
            {"text": "Ok.", "start": 67, "end": 70},  # len 3
            {"text": "Bye.", "start": 71, "end": 75},  # len 4
        ],
    ):
        result = semantic_chunker_instance.split_text(
            "This is a long sentence that already exceeds half min_chunk_size. Ok. Bye."
        )
        # The long sentence should not merge with "Ok." because its length is already > min_chunk_size/2
        # So each sentence should be its own chunk.
        assert len(result) == 3
        assert (
            result[0]["text"]
            == "This is a long sentence that already exceeds half min_chunk_size."
        )
        assert result[1]["text"] == "Ok."
        assert result[2]["text"] == "Bye."


def test_split_text_with_empy_sentences_from_splitter(semantic_chunker_instance):
    with patch.object(
        semantic_chunker_instance,
        "_split_sentences",
        return_value=[
            {"text": "Sentence 1.", "start": 0, "end": 11},
            {"text": "", "start": 11, "end": 11},  # Empty sentence
            {"text": "Sentence 2.", "start": 12, "end": 23},
        ],
    ):
        result = semantic_chunker_instance.split_text("Sentence 1..Sentence 2.")
        assert len(result) == 2
        assert result[0]["text"] == "Sentence 1."
        assert result[1]["text"] == "Sentence 2."


def test_split_text_with_consecutive_short_sentences_exceeding_max_chunk_size(
    semantic_chunker_instance,
):
    # min_chunk_size=10, max_chunk_size=50
    # Simulate a scenario where many short sentences merge, but the final merged chunk exceeds max_chunk_size
    with patch.object(
        semantic_chunker_instance,
        "_split_sentences",
        return_value=[
            {"text": "a.", "start": 0, "end": 2},
            {"text": "b.", "start": 3, "end": 5},
            {"text": "c.", "start": 6, "end": 8},
            {"text": "d.", "start": 9, "end": 11},
            {"text": "e.", "start": 12, "end": 14},
            {"text": "f.", "start": 15, "end": 17},
            {"text": "g.", "start": 18, "end": 20},
            {"text": "h.", "start": 21, "end": 23},
            {"text": "i.", "start": 24, "end": 26},
            {"text": "j.", "start": 27, "end": 29},
            {"text": "k.", "start": 30, "end": 32},
            {"text": "l.", "start": 33, "end": 35},
            {"text": "m.", "start": 36, "end": 38},
            {"text": "n.", "start": 39, "end": 41},
            {"text": "o.", "start": 42, "end": 44},
            {"text": "p.", "start": 45, "end": 47},
            {"text": "q.", "start": 48, "end": 50},
            {"text": "r.", "start": 51, "end": 53},
            {"text": "s.", "start": 54, "end": 56},
            {"text": "t.", "start": 57, "end": 59},
            {"text": "u.", "start": 60, "end": 62},
            {"text": "v.", "start": 63, "end": 65},
            {"text": "w.", "start": 66, "end": 68},
            {"text": "x.", "start": 69, "end": 71},
            {"text": "y.", "start": 72, "end": 74},
            {"text": "z.", "start": 75, "end": 77},
        ],
    ):
        # The _optimize_chunk_sizes should handle the max_chunk_size logic after initial short sentence merging
        result = semantic_chunker_instance.split_text(
            "a. b. c. d. e. f. g. h. i. j. k. l. m. n. o. p. q. r. s. t. u. v. w. x. y. z."
        )
        # Initial merging will happen for short sentences, then _optimize_chunk_sizes will split them if > max_chunk_size
        # The exact number of chunks will depend on how many sentences form a chunk that's < max_chunk_size
        # For simplicity, just check that it's not a single huge chunk and chunks are within limits
        assert len(result) > 1
        for chunk in result:
            assert (
                len(chunk["text"]) <= semantic_chunker_instance.max_chunk_size + 1
            )  # +1 for potential space
