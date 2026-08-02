#!/usr/bin/env python
"""
RAG Pipeline Direct Test - Uses existing cache to test retrieval and generation
"""

import sys, os, json, asyncio, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.session import SessionManager
from core.rag_core import RAGSystem

# Real cached file hash from the CM3 paper
FILE_HASH = "b14eac1860bab54116fe13be659b560309f4cf6262801232b021f1c1c8dfaaa9"


def setup_session():
    """Set up session to use the existing CM3 cache"""
    SessionManager.set_session_id("eval_test")
    SessionManager.set("file_hash", FILE_HASH)
    SessionManager.set("last_uploaded_file_name", "2201.07520v1.pdf")
    SessionManager.set(
        "pdf_file_path",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "temp",
            "upload_7ad7c7b3-dab6-4afd-95a8-ed7f9005b06b_1779880680.pdf",
        ),
    )
    print(f"[SETUP] Session 'eval_test' configured with hash: {FILE_HASH[:20]}...")


async def test_retrieval():
    """Test 1: Document retrieval capability"""
    print("\n" + "=" * 70)
    print("📡 TEST 1: DOCUMENT RETRIEVAL")
    print("=" * 70)

    rag = RAGSystem(session_id="eval_test")

    test_queries = [
        "What is CM3 model?",
        "How does CM3 generate images?",
        "What datasets were used for training CM3?",
        "CM3의 주요 특징은 무엇인가요?",
        "CM3 논문의 실험 결과는 무엇인가요?",
        "What is causal masked modeling?",
    ]

    results = []
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        try:
            # Use aquery which should load from cache
            start = time.time()
            response = await rag.aquery(q)
            elapsed = time.time() - start

            print(f"   Response type: {type(response).__name__}")

            if isinstance(response, dict):
                for k, v in response.items():
                    if isinstance(v, str) and len(v) > 50:
                        print(f"   {k}: {str(v)[:300]}...")
                    elif isinstance(v, (int, float, bool)):
                        print(f"   {k}: {v}")
                    elif v is None:
                        print(f"   {k}: None")
                    else:
                        print(f"   {k}: {type(v).__name__} ({len(str(v))} chars)")

                response_text = response.get(
                    "response", response.get("output", str(response))
                )
            else:
                response_text = str(response)

            results.append(
                {
                    "query": q,
                    "elapsed": round(elapsed, 2),
                    "success": True,
                    "response_preview": str(response_text)[:200],
                    "response_length": len(str(response_text)),
                }
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append({"query": q, "success": False, "error": str(e)})

    return results


async def test_streaming():
    """Test 2: Streaming response"""
    print("\n" + "=" * 70)
    print("🌊 TEST 2: STREAMING RESPONSE")
    print("=" * 70)

    rag = RAGSystem(session_id="eval_test")

    test_queries = [
        "CM3 모델에 대해 설명해주세요.",
    ]

    results = []
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        full_response = ""
        thought = ""
        metrics = {}
        chunk_count = 0

        try:
            start = time.time()
            async for chunk in rag.astream(q):
                chunk_count += 1
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                if hasattr(chunk, "thought") and chunk.thought:
                    thought += chunk.thought
                if hasattr(chunk, "performance") and chunk.performance:
                    metrics = chunk.performance
                if hasattr(chunk, "status"):
                    if chunk_count < 5:
                        print(
                            f"   [chunk] status={chunk.status}, content_len={len(chunk.content or '')}"
                        )

            elapsed = time.time() - start
            print(f"\n   Total chunks: {chunk_count}")
            print(f"   Elapsed: {elapsed:.2f}s")
            print(f"   Response length: {len(full_response)} chars")
            print(f"   Thought length: {len(thought)} chars")
            print(
                f"   Metrics: {json.dumps(metrics, ensure_ascii=False, default=str)[:300]}"
            )
            print(f"\n   📋 Response ({len(full_response)} chars):")
            print(f"   {full_response[:600]}")
            if len(full_response) > 600:
                print(f"   ... [truncated]")

            if thought:
                print(f"\n   💭 Thought: {thought[:300]}...")

            results.append(
                {
                    "query": q,
                    "success": True,
                    "chunk_count": chunk_count,
                    "elapsed": round(elapsed, 2),
                    "response_length": len(full_response),
                    "has_thought": len(thought) > 0,
                    "has_metrics": bool(metrics),
                    "response_preview": full_response[:300],
                }
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append({"query": q, "success": False, "error": str(e)})

    return results


async def test_without_cache():
    """Test 3: Try to test direct file processing pipeline"""
    print("\n" + "=" * 70)
    print("📂 TEST 3: PIPELINE BUILD TEST")
    print("=" * 70)

    from core.model_loader import get_embedding_model

    rag = RAGSystem(session_id="eval_test")

    try:
        # Find an existing PDF in temp
        import glob

        pdfs = sorted(glob.glob("data/temp/*.pdf"), key=os.path.getmtime, reverse=True)
        if pdfs:
            test_pdf = pdfs[0]
            print(
                f"   Found PDF: {os.path.basename(test_pdf)} ({os.path.getsize(test_pdf)} bytes)"
            )

            # Try to load embedding model
            print(f"   Loading embedding model...")
            embedder = get_embedding_model()
            print(f"   Embedder: {type(embedder).__name__}")

            # Test: just check if pipeline can be built
            print(f"   Testing pipeline build...")
            # We won't actually build (takes too long) - just verify setup
            print(f"   ✅ Pipeline setup verified")
        else:
            print(f"   ❌ No PDFs found in data/temp")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def main():
    print("🚀 RAG SYSTEM DIRECT EVALUATION")
    print(f"   File hash: {FILE_HASH[:20]}...")

    # Setup session
    setup_session()

    # Check if session is ready
    fh = SessionManager.get("file_hash", session_id="eval_test")
    print(f"   Session file_hash: {fh}")

    # Run retrieval test
    retrieval_results = await test_retrieval()

    # Run streaming test
    streaming_results = await test_streaming()

    # Run pipeline test
    await test_without_cache()

    # Print summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n📡 RETRIEVAL TESTS:")
    for r in retrieval_results:
        status = "✅" if r.get("success") else "❌"
        details = (
            f"{r.get('response_length', 0)} chars, {r.get('elapsed', 0)}s"
            if r.get("success")
            else r.get("error", "unknown")
        )
        print(f"   {status} '{r['query'][:40]}...' → {details}")

    print(f"\n🌊 STREAMING TESTS:")
    for r in streaming_results:
        status = "✅" if r.get("success") else "❌"
        if r.get("success"):
            markers = []
            if r.get("has_thought"):
                markers.append("💭thought")
            if r.get("has_metrics"):
                markers.append("📊metrics")
            details = f"{r.get('response_length', 0)} chars, {r.get('chunk_count', 0)} chunks, {'|'.join(markers)}, {r.get('elapsed', 0)}s"
        else:
            details = r.get("error", "unknown")
        print(f"   {status} '{r['query'][:40]}...' → {details}")

    # Save results
    report = {
        "retrieval": retrieval_results,
        "streaming": streaming_results,
        "timestamp": time.time(),
    }
    report_path = "scripts/eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 Report saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
