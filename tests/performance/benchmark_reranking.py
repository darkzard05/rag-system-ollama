"""
Reranking Performance Benchmark Script
Executes RAG pipeline with various top_k configurations to measure latency vs quality proxies.
"""

import asyncio
import csv
import os
import time
from unittest.mock import patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.common.config import AVAILABLE_EMBEDDING_MODELS, OLLAMA_MODEL_NAME
from src.core.rag_core import RAGSystem

# --- Test Configurations ---
EXPERIMENTS = [
    {"id": "Baseline", "retriever_k": 5, "reranker_enabled": False, "reranker_k": 0},
    {"id": "Exp-A", "retriever_k": 10, "reranker_enabled": True, "reranker_k": 3},
    {"id": "Exp-B", "retriever_k": 10, "reranker_enabled": True, "reranker_k": 5},
    {"id": "Exp-C", "retriever_k": 20, "reranker_enabled": True, "reranker_k": 5},
    {"id": "Exp-D", "retriever_k": 30, "reranker_enabled": True, "reranker_k": 7},
]

TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "List three key features mentioned.",
    "Summarize the conclusion in one sentence.",
]

# --- Mock Document Content ---
MOCK_DOC_TEXT = """
The RAG System (Retrieval-Augmented Generation) is designed for high performance.
It utilizes a modular architecture with core, api, and services packages.
Key features include Multi-tenant Session Isolation, Event-driven UI updates, and Adaptive Resource Control.
The system supports Distributed Search using an Interface-based architecture.
Performance optimization is achieved through AsyncIO and GPU batching.
The context window is set to 4096 tokens by default.
Reranking is performed using a Cross-Encoder model to improve relevance.
"""


async def run_benchmark():
    print("🚀 Starting Reranking Benchmark...")

    # 1. Setup Environment
    embedder = HuggingFaceEmbeddings(model_name=AVAILABLE_EMBEDDING_MODELS[0])
    rag_system = RAGSystem()

    # Manually load documents into RAGSystem (Skip PDF parsing for speed/stability)
    print("📝 Indexing mock documents...")
    docs = [
        Document(page_content=line, metadata={"page": i + 1, "source": "benchmark_doc"})
        for i, line in enumerate(MOCK_DOC_TEXT.split("\n"))
        if line.strip()
    ]

    vector_store = FAISS.from_documents(docs, embedder)
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5

    rag_system.vector_store = vector_store
    rag_system.bm25_retriever = bm25

    # Ensure retriever is built
    from core.rag_core import _create_ensemble_retriever

    rag_system.ensemble_retriever = _create_ensemble_retriever(vector_store, bm25)

    # 2. Run Experiments
    results = []

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        print(
            f"\n🧪 Running {exp_id} (Retriever k={exp['retriever_k']}, Rerank k={exp['reranker_k']})..."
        )

        # Patch Configurations
        retriever_config_patch = {
            "search_type": "similarity",
            "search_kwargs": {"k": exp["retriever_k"]},
            "ensemble_weights": [0.4, 0.6],
        }

        reranker_config_patch = {
            "enabled": exp["reranker_enabled"],
            "model_name": "BAAI/bge-reranker-v2-m3",
            "top_k": exp["reranker_k"],
            "max_rerank_docs": exp["retriever_k"],  # Rerank all retrieved
        }

        # Apply patches
        with (
            patch("src.core.rag_core.RETRIEVER_CONFIG", retriever_config_patch),
            patch("src.core.graph_builder.RERANKER_CONFIG", reranker_config_patch),
        ):
            # Rebuild graph to apply new configs
            from src.core.graph_builder import build_graph

            # We need to update the retriever's k dynamically
            rag_system.ensemble_retriever.retrievers[1].search_kwargs["k"] = exp[
                "retriever_k"
            ]  # FAISS
            rag_system.ensemble_retriever.retrievers[0].k = exp["retriever_k"]  # BM25

            rag_system.qa_chain = build_graph(retriever=rag_system.ensemble_retriever)

            for q_idx, question in enumerate(TEST_QUESTIONS):
                print(f"  - Q{q_idx + 1}: {question[:30]}...", end="", flush=True)

                start_time = time.perf_counter()
                try:
                    # Run Query
                    # Note: We need a real LLM for this to work. Assuming OLLAMA is running.
                    # If not, this will fail or hang.
                    from langchain_ollama import ChatOllama

                    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)

                    response = await rag_system.qa_chain.ainvoke(
                        {"input": question}, config={"configurable": {"llm": llm}}
                    )

                    duration = time.perf_counter() - start_time

                    # Metrics
                    answer_text = response.get("response", "")
                    docs = response.get("documents", [])
                    num_docs = len(docs)
                    answer_len = len(answer_text.split())

                    print(f" Done ({duration:.2f}s, {num_docs} docs)")

                    results.append(
                        {
                            "experiment_id": exp_id,
                            "question_id": q_idx + 1,
                            "retriever_k": exp["retriever_k"],
                            "reranker_k": exp["reranker_k"],
                            "latency_sec": round(duration, 3),
                            "answer_length_tokens": answer_len,
                            "num_docs_used": num_docs,
                            "reranker_enabled": exp["reranker_enabled"],
                        }
                    )

                except Exception as e:
                    print(f" Failed: {e}")
                    results.append(
                        {
                            "experiment_id": exp_id,
                            "question_id": q_idx + 1,
                            "error": str(e),
                        }
                    )

    # 3. Save Results
    output_file = "reports/RERANKING_BENCHMARK_RESULTS.csv"
    os.makedirs("reports", exist_ok=True)

    fieldnames = [
        "experiment_id",
        "question_id",
        "retriever_k",
        "reranker_k",
        "latency_sec",
        "answer_length_tokens",
        "num_docs_used",
        "reranker_enabled",
        "error",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Benchmark completed! Results saved to {output_file}")


if __name__ == "__main__":
    # Ensure src is in path
    import sys

    sys.path.append(os.path.join(os.getcwd(), "src"))

    asyncio.run(run_benchmark())
