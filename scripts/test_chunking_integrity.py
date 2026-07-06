import asyncio
import numpy as np
import sys
import os
from langchain_core.documents import Document

# Add src to path


from src.core.chunking import split_documents
from src.common.config import SEMANTIC_CHUNKER_CONFIG, TEXT_SPLITTER_CONFIG


class MockEmbedder:
    def embed_documents(self, texts):
        # Return random vectors based on text length for consistency
        return [np.random.rand(384).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(384).tolist()

    @property
    def model(self):
        return "mock-model"


async def test_metadata_integrity():
    # 1. Prepare mock documents
    docs = [
        Document(
            page_content="This is the first document content. It is quite long to ensure it gets split into multiple chunks if needed. "
            * 10,
            metadata={
                "file_path": "/path/to/doc1.pdf",
                "file_hash": "hash1",
                "page": 1,
            },
        ),
        Document(
            page_content="This is the second document content. It contains different information. "
            * 10,
            metadata={
                "file_path": "/path/to/doc2.pdf",
                "file_hash": "hash2",
                "page": 1,
            },
        ),
        Document(
            page_content="Third document content. Short and sweet. " * 5,
            metadata={
                "file_path": "/path/to/doc3.pdf",
                "file_hash": "hash3",
                "page": 1,
            },
        ),
    ]

    embedder = MockEmbedder()

    # Test cases: Standard splitting vs Semantic splitting
    test_cases = [
        ("Standard Splitting", False),
        ("Semantic Splitting", True),
    ]

    for name, use_semantic in test_cases:
        print(f"Testing {name}...")
        SEMANTIC_CHUNKER_CONFIG["enabled"] = use_semantic

        split_docs, _ = await split_documents(docs, embedder=embedder)

        # Verify every chunk has the correct metadata
        # Since we know the content, we can map it back
        errors = 0
        for i, chunk in enumerate(split_docs):
            content = chunk.page_content
            metadata = chunk.metadata

            # Simple check: Which source doc does this content belong to?
            found = False
            for source in docs:
                if content[:20] in source.page_content:
                    if (
                        metadata.get("file_path") != source.metadata["file_path"]
                        or metadata.get("file_hash") != source.metadata["file_hash"]
                    ):
                        print(
                            f"Metadata Mismatch in chunk {i}! Expected {source.metadata}, got {metadata}"
                        )
                        errors += 1
                    found = True
                    break
            if not found:
                print(f"Chunk {i} content not found in any source document.")
                errors += 1

        if errors == 0:
            print(
                f"SUCCESS: {name} metadata integrity verified. ({len(split_docs)} chunks)"
            )
        else:
            print(f"FAILURE: {name} metadata integrity failed with {errors} errors.")


async def main():
    await test_metadata_integrity()


if __name__ == "__main__":
    asyncio.run(main())
