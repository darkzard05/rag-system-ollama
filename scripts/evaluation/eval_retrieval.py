import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from src.core.rag_core import RAGSystem
from src.core.model_loader import ModelManager
from src.core.session import SessionManager
from src.common.config import DEFAULT_EMBEDDING_MODEL
from src.common.logging_config import setup_logging


async def run_retrieval_eval():
    setup_logging(log_level="INFO")

    print("\n" + "=" * 50)
    print("[E2E] RAG Retrieval Evaluation (LLM-less)")
    print("=" * 50)

    # 1. 세션 및 RAG 초기화
    session_id = f"retrieval-eval-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    # 2. 모델 준비
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

    # 3. 파이프라인 구축
    file_name = os.path.basename(test_pdf)
    await rag.build_pipeline(test_pdf, file_name, embedder)

    # 4. 리트리버 준비 (루프 외부에서 한 번만 수행)
    try:
        from core.retriever_factory import create_vector_store
        from core.document_processor import load_pdf_docs
        from core.chunking import split_documents

        docs = load_pdf_docs(test_pdf, file_name)
        doc_splits, vectors = await split_documents(docs, embedder=embedder)
        vector_store = create_vector_store(doc_splits, embedder, vectors=vectors)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("   Retriever ready.")
    except Exception as e:
        print(f"❌ Error preparing retriever: {e}")
        return

    # 5. 테스트 쿼리 셋
    queries = [
        "What is the primary objective of the CM3 model?",
        "How does the tokenization process for images work in CM3?",
        "What are the differences between CM3 and DALL-E?",
        "Explain the multi-modal alignment mechanism used in CM3.",
        "What are the key components of the encoder-decoder structure?",
        "Which datasets were used for training CM3?",
        "How is the loss function formulated in the paper?",
        "What are the main findings of the quantitative evaluation?",
        "How does the model handle high-resolution images?",
        "What are the limitations of the CM3 approach?",
    ]

    results = []

    for i, query in enumerate(queries):
        print(f"   [{i + 1}/{len(queries)}] Retrieving context for: '{query[:50]}...'")

        start_t = asyncio.get_event_loop().time()

        try:
            relevant_docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in relevant_docs])
        except Exception as e:
            print(f"   ❌ Error: {e}")
            context = "ERROR"
            relevant_docs = []

        q_time = asyncio.get_event_loop().time() - start_t

        results.append(
            {
                "query": query,
                "context": context,
                "time": q_time,
                "doc_count": len(relevant_docs),
            }
        )

    # 결과 저장
    with open("retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"Retrieval Evaluation Finished. Results saved to retrieval_results.json")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(run_retrieval_eval())
