import pytest

pytestmark = pytest.mark.skip(reason="Functionality removed/refactored")
from langchain_core.documents import Document


def test_merge_entity_docs_empty():
    """Case 1: Empty dictionaries"""
    assert merge_entity_docs({}, {}) == {}


def test_merge_entity_docs_one_empty():
    """Case 2: One empty dictionary"""
    doc1 = Document(page_content="doc1")
    left = {"entity1": [doc1]}
    assert merge_entity_docs(left, {}) == {"entity1": [doc1]}
    assert merge_entity_docs({}, left) == {"entity1": [doc1]}


def test_merge_entity_docs_disjoint_keys():
    """Case 3: Disjoint keys"""
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    left = {"entity1": [doc1]}
    right = {"entity2": [doc2]}
    expected = {"entity1": [doc1], "entity2": [doc2]}
    assert merge_entity_docs(left, right) == expected


def test_merge_entity_docs_overlapping_keys():
    """Case 4: Overlapping keys"""
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    doc3 = Document(page_content="doc3")
    left = {"entity1": [doc1]}
    right = {"entity1": [doc2], "entity2": [doc3]}
    expected = {"entity1": [doc1, doc2], "entity2": [doc3]}
    assert merge_entity_docs(left, right) == expected


def test_merge_entity_docs_immutability():
    """Case 5: Verify no mutation of original dicts"""
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    left = {"entity1": [doc1]}
    right = {"entity1": [doc2]}
    _ = merge_entity_docs(left, right)
    assert left == {"entity1": [doc1]}
    assert right == {"entity1": [doc2]}
