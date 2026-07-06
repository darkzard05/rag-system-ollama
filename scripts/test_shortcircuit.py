import asyncio
import sys
import os
from dataclasses import dataclass, field
from typing import Any, List

# 프로젝트 루트 경로 추가


from langchain_core.documents import Document
from src.core.graph_builder import (
    retrieve_and_rerank,
    grade_documents,
    GraphState,
    RunnableConfig,
)
from src.core.reranker import DistributedReranker, RerankerStrategy
from src.common.config import CONFIDENCE_THRESHOLD


@dataclass
class MockState:
    input: str = ""
    intent: str = "rag"
    is_cached: bool = False
    search_weights: dict = None
    retry_count: int = 0
    relevant_docs: List[Document] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)

    def get(self, key, default=None):
        return getattr(self, key, default)


def get_state_attr(state, key, default=None):
    return getattr(state, key, default)


async def test_reranker_and_shortcircuit():
    print("\n" + "=" * 60)
    print("BGE Reranker & Short-Circuit Integration Test")
    print("=" * 60)

    # 1. Mock Configuration
    # We mock the configurable part of the config
    config = {
        "configurable": {
            "bm25_retriever": None,  # We will mock the results in the state
            "faiss_retriever": None,
            "num_ctx": 4096,
            "llm": None,  # Not needed for short-circuit
        }
    }

    # Since retrieve_and_rerank depends on retrievers, we'll mock its output
    # and test grade_documents directly for the short-circuit.

    # Case 1: High Confidence (Should trigger la Short-circuit -> generate)
    print("\n[Case 1] High Confidence Document")
    state_high = MockState(
        input="What is the battery capacity of iPhone 15?",
        relevant_docs=[
            Document(
                page_content="The battery capacity of iPhone 15 is 3,349mAh.",
                metadata={"rerank_score": 0.98, "page": 1},
            ),
            Document(
                page_content="Other phone info...",
                metadata={"rerank_score": 0.1, "page": 2},
            ),
        ],
    )

    # Note: grade_documents expects GraphState, which is a Pydantic model usually.
    # We use a simple object that mimics it.
    # we need to monkeypatch get_state_attr if the original one is used.

    # Since I can't easily monkeypatch the function in the module without editing the file,
    # I'll just call it and see if it works with my MockState.
    # Actually, graph_builder.py uses get_state_attr.

    try:
        # We need to pass a writer (None is okay)
        result_high = await grade_documents(state_high, config, writer=None)
        print(
            f"Result: {result_high} -> {'✅ PASS' if result_high.get('intent') == 'generate' else '❌ FAIL'}"
        )
    except Exception as e:
        print(f"❌ Error: {e}")

    # Case 2: Low Confidence (Should trigger Short-circuit -> transform/rewrite)
    print("\n[Case 2] Low Confidence Document")
    state_low = MockState(
        input="What is the battery capacity of iPhone 15?",
        relevant_docs=[
            Document(
                page_content="The weather today is sunny.",
                metadata={"rerank_score": -2.0, "page": 1},
            )
        ],
    )
    try:
        result_low = await grade_documents(state_low, config, writer=None)
        print(
            f"Result: {result_low} -> {'✅ PASS' if result_low.get('intent') == 'transform' else '❌ FAIL'}"
        )
    except Exception as e:
        print(f"❌ Error: {e}")

    # Case 3: Medium Confidence (Should trigger LLM grading - but will fail because LLM is None)
    print("\n[Case 3] Medium Confidence Document (LLM Fallback)")
    state_med = MockState(
        input="What is the battery capacity of iPhone 15?",
        relevant_docs=[
            Document(
                page_content="Some ambiguous info about batteries.",
                metadata={"rerank_score": 0.4, "page": 1},
            )
        ],
    )
    try:
        result_med = await grade_documents(state_med, config, writer=None)
        print(
            f"Result: {result_med} -> {'✅ PASS' if result_med.get('intent') == 'generate' else '❌ FAIL'}"
        )
        print(
            "(Note: LLM is None, so it should fall back to 'generate' as per the code's exception handler)"
        )
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_reranker_and_shortcircuit())
