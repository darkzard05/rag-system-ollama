import asyncio

# Add src to python path
import numpy as np
from langchain_core.documents import Document
from src.common.constants import DOC_JOIN_DELIMITER
from src.core.chunking import _postprocess_metadata


class MockEmbedder:
    def embed_documents(self, texts):
        return [np.random.rand(128).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(128).tolist()


async def test_offset_integrity():
    print("\n--- Testing Offset Integrity ---")
    docs = [
        Document(
            page_content="Content of Page 1",
            metadata={"page": 1, "file_path": "file1.pdf", "file_hash": "h1"},
        ),
        Document(
            page_content="Content of Page 2",
            metadata={"page": 2, "file_path": "file1.pdf", "file_hash": "h1"},
        ),
        Document(
            page_content="Content of Page 3",
            metadata={"page": 3, "file_path": "file2.pdf", "file_hash": "h2"},
        ),
    ]

    # We use a small chunk size to ensure we get multiple chunks

    # Temporarily force a simple splitter for testing

    # We want to test _postprocess_metadata specifically
    # Create some mock split docs with start_index
    split_docs = [
        Document(page_content="Content of Page 1", metadata={"start_index": 0}),
        Document(
            page_content="Content of Page 2",
            metadata={
                "start_index": len("Content of Page 1") + len(DOC_JOIN_DELIMITER)
            },
        ),
        Document(
            page_content="Content of Page 3",
            metadata={
                "start_index": len("Content of Page 1")
                + len(DOC_JOIN_DELIMITER)
                + len("Content of Page 2")
                + len(DOC_JOIN_DELIMITER)
            },
        ),
    ]

    _postprocess_metadata(split_docs, docs)

    results = []
    for _, doc in enumerate(split_docs):
        results.append((doc.metadata.get("file_path"), doc.metadata.get("file_hash")))

    expected = [("file1.pdf", "h1"), ("file1.pdf", "h1"), ("file2.pdf", "h2")]
    if results == expected:
        print("✅ Offset Integrity: PASSED")
    else:
        print(f"❌ Offset Integrity: FAILED. Expected {expected}, got {results}")


async def test_section_recognition():
    print("\n--- Testing Section Recognition ---")
    # Create documents that mimic a research paper structure
    docs = [
        Document(
            page_content="This is the main body content. It should be content.",
            metadata={"page": 1},
        ),
        Document(
            page_content="## References\nRef 1: Paper A\nRef 2: Paper B",
            metadata={"page": 2},
        ),
        Document(
            page_content="## Appendix\nThis is the appendix. It should be content again.",
            metadata={"page": 3},
        ),
    ]

    # We will mock the splitting process to just get these docs as chunks
    split_docs = [
        Document(
            page_content="This is the main body content. It should be content.",
            metadata={"page": 1},
        ),
        Document(
            page_content="## References\nRef 1: Paper A\nRef 2: Paper B",
            metadata={"page": 2},
        ),
        Document(
            page_content="## Appendix\nThis is the appendix. It should be content again.",
            metadata={"page": 3},
        ),
    ]

    _postprocess_metadata(split_docs, docs)

    # Check flags
    # Doc 0: Content
    # Doc 1: Reference
    # Doc 2: Content (Appendix)

    results = []
    for doc in split_docs:
        results.append(
            (doc.metadata.get("is_reference"), doc.metadata.get("is_content"))
        )

    expected = [
        (False, True),  # Body
        (True, False),  # References
        (False, True),  # Appendix
    ]

    if results == expected:
        print("✅ Section Recognition: PASSED")
    else:
        print(f"❌ Section Recognition: FAILED. Expected {expected}, got {results}")


async def main():
    await test_offset_integrity()
    await test_section_recognition()


if __name__ == "__main__":
    asyncio.run(main())
