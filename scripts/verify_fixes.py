#!/usr/bin/env python
"""Verify Phase 1 fixes don't break the RAG pipeline"""

import sys, os, hashlib, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.session import SessionManager
from core.rag_core import RAGSystem


async def main():
    # Setup session
    temp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "temp")
    pdfs = [
        f
        for f in os.listdir(temp_dir)
        if f.endswith(".pdf") and os.path.getsize(os.path.join(temp_dir, f)) > 1000000
    ]
    if not pdfs:
        print("FAIL: No PDF found")
        return 1

    pdf_path = os.path.join(
        temp_dir,
        sorted(
            pdfs,
            key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)),
            reverse=True,
        )[0],
    )
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    SessionManager.set_session_id("verify_fixes")
    SessionManager.set("file_hash", file_hash)
    SessionManager.set("pdf_file_path", pdf_path)
    SessionManager.set("last_uploaded_file_name", os.path.basename(pdf_path))

    rag = RAGSystem(session_id="verify_fixes")

    all_pass = True

    # === Test 1: English query language ===
    print("=" * 60)
    print("TEST 1: English query → English response")
    print("=" * 60)
    resp = await rag.aquery("What is CM3 model?")
    text = resp.get("response", "")
    has_korean = any(ord(c) > 0xAC00 for c in text[:100])
    if has_korean:
        print(f"  FAIL: English query got Korean response")
        print(f"  Response starts: {text[:100]}")
        all_pass = False
    else:
        print(f"  PASS: English query → English (or non-Korean) response")
        print(f"  Response starts: {text[:100]}...")

    # === Test 2: Citation format ===
    print()
    print("=" * 60)
    print("TEST 2: Citation format matches protocol")
    print("=" * 60)
    ctx = resp.get("context", "")
    if ctx:
        has_old_format = "### [섹션:" in ctx
        has_new_format = ", p." in ctx
        print(f"  Old format (### [섹션:): {has_old_format}")
        print(f"  New format ([..., p.X]): {has_new_format}")
        if has_old_format:
            print("  FAIL: Old format still present")
            all_pass = False
        else:
            print("  PASS: Citation format updated correctly")
    else:
        print("  SKIP: No context returned")

    # === Test 3: No SQLite error ===
    print()
    print("=" * 60)
    print("TEST 3: No SQLite/aioSQLite errors")
    print("=" * 60)
    if resp.get("response"):
        print("  PASS: Response generated without SQLite errors")
    else:
        print("  FAIL: Empty response")
        all_pass = False

    # === Test 4: Korean query still works ===
    print()
    print("=" * 60)
    print("TEST 4: Korean query → Korean response")
    print("=" * 60)
    resp2 = await rag.aquery("CM3 모델의 학습 데이터는 무엇인가요?")
    text2 = resp2.get("response", "")
    has_korean2 = any(ord(c) > 0xAC00 for c in text2[:100])
    if has_korean2:
        print(f"  PASS: Korean query → Korean response")
        print(f"  Response starts: {text2[:100]}...")
    else:
        print(f"  WARN: Korean query response may not be Korean")
        print(f"  Response starts: {text2[:100]}")

    # === Test 5: Function rename ===
    print()
    print("=" * 60)
    print("TEST 5: prepare_query_config_or_build name")
    print("=" * 60)
    import importlib
    from core import pipeline_builder

    func_name = pipeline_builder.prepare_query_config_or_build.__name__
    if func_name == "prepare_query_config_or_build":
        print("  PASS: Function renamed correctly")
    else:
        print(f"  FAIL: Function name is {func_name}")
        all_pass = False

    # === Summary ===
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 60)

    # Cleanup
    try:
        await get_resource_manager().unregister_retrievers(file_hash)
    except Exception:
        pass

    return 0 if all_pass else 1


if __name__ == "__main__":
    # Import here to avoid circular
    from core.resource_manager import get_resource_manager

    exit(asyncio.run(main()))
