from langchain_core.documents import Document
from rag_core import VectorStoreCache


def test_vector_store_cache_serialization():
    """Test serialization and deserialization of documents."""
    # 1. Arrange: Create dummy documents and a cache instance
    docs = [
        Document(page_content="Hello world", metadata={"page": 1}),
        Document(
            page_content="Test document", metadata={"page": 2, "source": "test.pdf"}
        ),
    ]

    # The __init__ requires bytes and model name, but they are not used by the methods under test.
    # We can pass dummy values.
    cache = VectorStoreCache(file_bytes=b"dummy", embedding_model_name="dummy-model")

    # 2. Act: Serialize the documents
    serialized_docs = cache._serialize_docs(docs)

    # 3. Assert: Check the serialized format
    assert isinstance(serialized_docs, list)
    assert len(serialized_docs) == 2
    assert serialized_docs[0] == {
        "page_content": "Hello world",
        "metadata": {"page": 1},
    }
    assert serialized_docs[1] == {
        "page_content": "Test document",
        "metadata": {"page": 2, "source": "test.pdf"},
    }

    # 4. Act: Deserialize the documents
    deserialized_docs = cache._deserialize_docs(serialized_docs)

    # 5. Assert: Check if they match the original
    assert isinstance(deserialized_docs, list)
    assert len(deserialized_docs) == 2
    # langchain Document objects are compared by their content and metadata dicts
    assert deserialized_docs[0].page_content == docs[0].page_content
    assert deserialized_docs[0].metadata == docs[0].metadata
    assert deserialized_docs[1].page_content == docs[1].page_content
    assert deserialized_docs[1].metadata == docs[1].metadata
