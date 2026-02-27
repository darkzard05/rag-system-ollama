from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document
from common.utils import (
    apply_tooltips_to_response,
    clean_query_text,
    preprocess_text,
)

# Optimized functions
from core.graph_builder import _merge_consecutive_chunks
from core.document_processor import load_pdf_docs
from core.semantic_chunker import (
    EmbeddingBasedSemanticChunker,
)


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
    assert _merge_consecutive_chunks([doc])[0].page_content == "Content"


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
    # In my restored version, it returns copies, so we compare content
    result = _merge_consecutive_chunks(docs)
    assert len(result) == 3
    assert result[0].page_content == "Content 1"
    assert result[1].page_content == "Content 2"
    assert result[2].page_content == "Content 3"


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
    result = _merge_consecutive_chunks(docs)
    assert len(result) == 1
    assert result[0].page_content == "Part 1. Part 2. Part 3."
    assert result[0].metadata["chunk_index"] == 2


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


# --- Test preprocess_text (src/common/utils.py) ---
def test_preprocess_text_empty():
    assert preprocess_text("") == ""


def test_preprocess_text_multiple_spaces():
    assert preprocess_text("  Hello   World  ") == "Hello World"


# --- Test clean_query_text (src/common/utils.py) ---
def test_clean_query_text_with_prefixes():
    assert clean_query_text("1. This is a query.") == "This is a query."
    assert clean_query_text("Query: What is it?") == "What is it?"


# --- Test load_pdf_docs (src/core/document_processor.py) ---
@patch("pymupdf4llm.to_markdown")
@patch("core.session.SessionManager")
def test_load_pdf_docs_success(mock_session_manager, mock_to_markdown):
    # Mock pymupdf4llm output
    mock_to_markdown.return_value = [
        {"text": "Page 1 text", "metadata": {"page": 1, "page_count": 2}, "words": []},
        {"text": "Page 2 text", "metadata": {"page": 2, "page_count": 2}, "words": []},
    ]

    docs = load_pdf_docs("dummy.pdf", "dummy.pdf")
    assert len(docs) == 2
    assert docs[0].page_content == "Page 1 text"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert docs[0].metadata["source"] == "dummy.pdf"


# --- Test EmbeddingBasedSemanticChunker (src/core/semantic_chunker.py) ---
@pytest.fixture
def semantic_chunker_instance():
    return EmbeddingBasedSemanticChunker(
        embedder=MockEmbedder(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_value=75.0,
        min_chunk_size=10,
        max_chunk_size=50,
        similarity_threshold=0.5,
        batch_size=2,
    )


def test_find_breakpoints_multiple_breakpoints(semantic_chunker_instance):
    semantic_chunker_instance.breakpoint_threshold_type = "similarity_threshold"
    semantic_chunker_instance.breakpoint_threshold_value = 0.5
    similarities = [0.8, 0.2, 0.7, 0.3, 0.9]
    # Breakpoints are indices where similarity < threshold
    # 0.2 is at index 1, 0.3 is at index 3
    # The function returns [2, 4] (1-based positions for splitting)
    assert semantic_chunker_instance._find_breakpoints(similarities) == [2, 4]
