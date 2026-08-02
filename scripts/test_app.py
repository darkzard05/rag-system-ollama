#!/usr/bin/env python
"""
Test the running Streamlit app by uploading a PDF and sending queries.
Uses the same SessionManager as the running app.
"""

import sys, os, hashlib, json, asyncio, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.session import SessionManager
from core.rag_core import RAGSystem


def find_cm3_pdf():
    """Find a CM3 PDF in data/temp with real content"""
    temp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "temp")
    pdfs = [f for f in os.listdir(temp_dir) if f.endswith(".pdf")]
    # Real CM3 PDFs are ~5.5MB
    real_pdfs = [
        f for f in pdfs if os.path.getsize(os.path.join(temp_dir, f)) > 1000000
    ]
    # Sort by newest first
    real_pdfs.sort(
        key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True
    )
    if real_pdfs:
        return os.path.join(temp_dir, real_pdfs[0])
    return None


def compute_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_cache_for_hash(file_hash):
    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", ".model_cache", "vector_store_cache"
    )
    if not os.path.isdir(cache_dir):
        return None
    for d in os.listdir(cache_dir):
        if file_hash in d:
            return os.path.join(cache_dir, d)
    return None


def setup_and_test():
    """Test with a real PDF file"""
    print("=" * 60)
    print("RAG SYSTEM - PRACTICAL TEST")
    print("=" * 60)

    # 1. Find PDF
    pdf_path = find_cm3_pdf()
    if not pdf_path:
        print("❌ No CM3 PDF found in data/temp")
        return
    print(
        f"📄 Using PDF: {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path)} bytes)"
    )

    # 2. Compute hash
    file_hash = compute_hash(pdf_path)
    print(f"🔑 SHA256: {file_hash[:32]}...")

    # 3. Check if this hash is cached
    cache_dir = check_cache_for_hash(file_hash)
    if cache_dir:
        print(f"✅ Cache found: {os.path.basename(cache_dir)[:50]}...")
        # List cache contents
        for f in sorted(os.listdir(cache_dir)):
            fp = os.path.join(cache_dir, f)
            if os.path.isdir(fp):
                sub = os.listdir(fp)
                total_size = sum(os.path.getsize(os.path.join(fp, sf)) for sf in sub)
                print(f"   [DIR] {f}/ ({len(sub)} files, {total_size / 1024:.0f} KB)")
            else:
                print(f"   {f} ({os.path.getsize(fp)} bytes)")
    else:
        print("⚠️ No cache for this hash - pipeline will need to build from scratch")

    # 4. Test SessionManager with this file
    print(f"\n--- Session Setup ---")
    SessionManager.set_session_id("app_test")
    SessionManager.set("file_hash", file_hash)
    SessionManager.set("pdf_file_path", pdf_path)
    SessionManager.set("last_uploaded_file_name", os.path.basename(pdf_path))

    fh = SessionManager.get("file_hash", session_id="app_test")
    print(f"Session file_hash set: {fh[:20] if fh else 'None'}...")

    # 5. Try to initialize RAGSystem and check pipeline status
    print(f"\n--- RAGSystem Test ---")

    async def test():
        rag = RAGSystem(session_id="app_test")

        # Test queries
        test_queries = [
            "CM3 모델에 대해 설명해주세요.",
            "What is CM3 and how does it work?",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 50)
            try:
                start = time.time()
                response = await rag.aquery(query)
                elapsed = time.time() - start

                if isinstance(response, dict):
                    # Pretty print
                    resp_text = response.get("response", response.get("output", ""))
                    thought = response.get("thought", "")
                    perf = response.get("performance", {})

                    print(f"✅ Response ({len(resp_text)} chars, {elapsed:.2f}s):")
                    print(resp_text[:600])
                    if len(resp_text) > 600:
                        print(f"... [truncated, total {len(resp_text)} chars]")

                    if thought:
                        print(f"\n💭 Thought: {thought[:200]}...")

                    if perf:
                        print(f"\n📊 Metrics:")
                        for k, v in perf.items():
                            print(f"   {k}: {v}")
                else:
                    print(f"Response: {str(response)[:500]}")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()

    asyncio.run(test())


if __name__ == "__main__":
    setup_and_test()
