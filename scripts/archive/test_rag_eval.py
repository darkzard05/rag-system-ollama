#!/usr/bin/env python
"""
RAG System Evaluation Script
Tests the full pipeline: retrieval, generation, streaming, and output quality.
"""

import sys, os, json, asyncio, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.rag_core import RAGSystem


async def test_retrieval(rag: RAGSystem):
    """Test 1: Document retrieval quality"""
    print("\n" + "=" * 70)
    print("📡 TEST 1: DOCUMENT RETRIEVAL QUALITY")
    print("=" * 70)

    ctx = rag.get_context()
    retrievers = ctx.get("retrievers", {})
    if not retrievers:
        print("❌ No retrievers available")
        return []

    rid = list(retrievers.keys())[0]
    retriever = retrievers[rid]
    results = []

    test_queries = [
        # English
        "What is CM3 and how does it work?",
        "What datasets were used to train CM3?",
        "How does CM3 generate images?",
        "What is causal masked modeling?",
        "What were the results of CM3 experiments?",
        # Korean
        "CM3 모델이 무엇인가요?",
        "CM3는 어떻게 이미지를 생성하나요?",
        "CM3의 학습에는 어떤 데이터셋이 사용되었나요?",
        "CM3와 DALL-E의 차이점은 무엇인가요?",
        "CM3 논문의 주요 실험 결과는 무엇인가요?",
    ]

    for q in test_queries:
        print(f"\n📝 Query: {q}")
        try:
            start = time.time()
            docs = await retriever._aretrieve(q)
            elapsed = time.time() - start

            print(f"   Retrieved {len(docs)} docs in {elapsed:.2f}s")
            doc_info = []
            for i, d in enumerate(docs[:3]):
                content = (
                    d.page_content[:200] if hasattr(d, "page_content") else str(d)[:200]
                )
                score = d.metadata.get("score", 0) if hasattr(d, "metadata") else 0
                source = (
                    d.metadata.get("source", "N/A") if hasattr(d, "metadata") else "N/A"
                )
                page = d.metadata.get("page", "?") if hasattr(d, "metadata") else "?"
                print(f"   [{i + 1}] score={score:.4f}, page={page}: {content}...")
                doc_info.append(
                    {"score": score, "page": page, "content_preview": content[:100]}
                )

            results.append(
                {
                    "query": q,
                    "num_docs": len(docs),
                    "elapsed": elapsed,
                    "top_docs": doc_info,
                    "has_relevant": any(d.get("score", 0) > 0.5 for d in doc_info)
                    if doc_info
                    else False,
                }
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({"query": q, "error": str(e)})

    return results


async def test_generation(rag: RAGSystem):
    """Test 2: Full query → answer generation (streaming)"""
    print("\n" + "=" * 70)
    print("🧠 TEST 2: FULL QA GENERATION")
    print("=" * 70)

    test_queries = [
        "CM3 모델의 주요 특징을 설명해주세요.",
        "CM3는 어떻게 기존 모델과 다른가요?",
    ]

    results = []
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        print(f"{'─' * 60}")
        try:
            start = time.time()
            full_response = ""
            thought_content = ""
            metrics = {}

            async for chunk in rag.astream(q):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                if hasattr(chunk, "thought") and chunk.thought:
                    thought_content += chunk.thought
                if hasattr(chunk, "performance") and chunk.performance:
                    metrics = chunk.performance
                if hasattr(chunk, "status") and chunk.status:
                    pass  # just status message

            elapsed = time.time() - start

            # Print results
            print(f"\n📋 Response ({len(full_response)} chars):")
            print(full_response[:500])
            if len(full_response) > 500:
                print(f"... [truncated, total {len(full_response)} chars]")

            if thought_content:
                print(
                    f"\n💭 Thought ({len(thought_content)} chars): {thought_content[:200]}..."
                )

            if metrics:
                print(
                    f"\n📊 Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}"
                )

            print(f"\n⏱️  Total time: {elapsed:.2f}s")

            results.append(
                {
                    "query": q,
                    "response_length": len(full_response),
                    "elapsed": elapsed,
                    "has_content": len(full_response) > 0,
                    "has_citations": "[p." in full_response or "p." in full_response,
                    "has_korean": any(ord(c) > 0xAC00 for c in full_response)
                    if full_response
                    else False,
                    "metrics": metrics,
                }
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append({"query": q, "error": str(e)})

    return results


async def test_direct_retrieval_no_graph(rag: RAGSystem):
    """Test 3: Direct retriever response without full graph"""
    print("\n" + "=" * 70)
    print("🔍 TEST 3: DIRECT RAG RESPONSE (NO GRAPH)")
    print("=" * 70)

    try:
        start = time.time()
        response = await rag.aquery("CM3 모델의 학습 방법을 설명해주세요.")
        elapsed = time.time() - start

        print(f"\n📋 Direct Response ({elapsed:.2f}s):")

        if isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, str) and len(v) > 50:
                    print(f"  {k}: {v[:300]}...")
                else:
                    print(f"  {k}: {v}")
        else:
            print(str(response)[:500])

        return {
            "type": str(type(response).__name__),
            "elapsed": elapsed,
            "has_response": bool(response),
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


async def main():
    print("🚀 RAG System Evaluation")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Initializing RAGSystem...")
    rag = RAGSystem()
    await rag.initialize()
    print("[OK] RAGSystem initialized")

    # Check existing state
    ctx = rag.get_context()
    retrievers = ctx.get("retrievers", {})
    config = ctx.get("config", {})
    print(f"\n[INFO] Retrievers: {len(retrievers)}")
    print(f"[INFO] Config available: {bool(config)}")

    if retrievers:
        for rid in retrievers:
            print(f"  - Retriever ID: {rid[:40]}...")

    # Run tests
    retrieval_results = await test_retrieval(rag)
    generation_results = await test_generation(rag)
    direct_results = await test_direct_retrieval_no_graph(rag)

    # Summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)

    # Retrieval summary
    num_retrieval = len(retrieval_results)
    num_retrieval_ok = sum(
        1 for r in retrieval_results if not r.get("error") and r.get("has_relevant")
    )
    print(
        f"\n📡 Retrieval: {num_retrieval_ok}/{num_retrieval} queries had relevant results"
    )

    # Generation summary
    for r in generation_results:
        if "error" in r:
            print(f"❌ Generation '{r['query'][:30]}...': ERROR - {r['error']}")
        else:
            markers = []
            if r.get("has_content"):
                markers.append("✅ has_content")
            if r.get("has_citations"):
                markers.append("📎 citations")
            if r.get("has_korean"):
                markers.append("🇰🇷 korean")
            print(
                f"✅ Generation '{r['query'][:30]}...': {', '.join(markers)} ({r['response_length']} chars, {r['elapsed']:.1f}s)"
            )

    # Save detailed results
    report = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "direct": direct_results,
    }
    report_path = "scripts/eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 Detailed report saved: {report_path}")

    await rag.cleanup()
    print("\n✅ Evaluation complete")


if __name__ == "__main__":
    asyncio.run(main())
