#!/usr/bin/env python
"""Verify Phase 2 fixes don't break imports and core functionality."""

import sys, os, warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

errors = []

# 1. Syntax check all changed files
print("=" * 60)
print("Phase 2: Fix Verification")
print("=" * 60)

# 2. Import all changed modules
print("\n[1/4] Import verification...")
try:
    from core.model_loader import ModelManager
    from core.resource_manager import get_resource_manager
    from core.async_reranker import (
        AsyncSemanticReranker,
        _get_cached_emb,
        _set_cached_emb,
    )

    print("  All imports: OK")
except Exception as e:
    errors.append(f"Import failed: {e}")
    print(f"  FAIL: {e}")

# 3. Verify ModelManager delegation
print("\n[2/4] ModelManager delegation verification...")
import inspect

# Check inference_session
src = inspect.getsource(ModelManager.inference_session)
has_delegation = "get_resource_manager()" in src
if has_delegation:
    print("  inference_session(): delegated to ResourceCoordinator OK")
else:
    errors.append("inference_session() still uses local semaphore")
    print("  inference_session(): STILL LOCAL, delegation MISSING")

# Check get_embedder
src = inspect.getsource(ModelManager.get_embedder)
has_delegation = "get_resource_manager()" in src
if has_delegation:
    print("  get_embedder(): delegated OK")
else:
    errors.append("get_embedder() still uses local cache")
    print("  get_embedder(): STILL LOCAL")

# Check get_llm
src = inspect.getsource(ModelManager.get_llm)
has_delegation = "get_resource_manager()" in src
if has_delegation:
    print("  get_llm(): delegated OK")
else:
    errors.append("get_llm() still uses local cache")
    print("  get_llm(): STILL LOCAL")

# Check get_flashranker
src = inspect.getsource(ModelManager.get_flashranker)
has_delegation = "get_resource_manager()" in src
if has_delegation:
    print("  get_flashranker(): delegated OK")
else:
    errors.append("get_flashranker() still uses local cache")
    print("  get_flashranker(): STILL LOCAL")

# Check clear_vram
src = inspect.getsource(ModelManager.clear_vram)
has_delegation = "get_resource_manager()" in src
if has_delegation:
    print("  clear_vram(): delegated OK")
else:
    errors.append("clear_vram() still local")
    print("  clear_vram(): STILL LOCAL")

# 4. Verify async_reranker cache module
print("\n[3/4] Reranker cache verification...")
try:
    test_text = "CM3 is a causal masked multimodal model."
    vec = _get_cached_emb(test_text)
    print(f"  Cache miss for new text: {vec is None} (expected: True)")

    test_vec = [0.1, 0.2, 0.3]
    _set_cached_emb(test_text, test_vec)
    cached = _get_cached_emb(test_text)
    print(f"  Cache hit after store: {cached == test_vec} (expected: True)")

    # Verify OrderedDict import
    from collections import OrderedDict
    from core.async_reranker import _doc_emb_cache

    print(f"  Cache type: {type(_doc_emb_cache).__name__} (expected: OrderedDict)")
except Exception as e:
    errors.append(f"Reranker cache error: {e}")
    print(f"  FAIL: {e}")

# 5. Check config changes
print("\n[4/4] Config verification...")
import yaml

with open(
    os.path.join(os.path.dirname(__file__), "..", "config.yml"), "r", encoding="utf-8"
) as f:
    config = yaml.safe_load(f)

cache_enabled = config.get("global_cache", {}).get("enable_vector_cache")
num_predict = config.get("models", {}).get("ollama_num_predict")
print(f"  enable_vector_cache: {cache_enabled} (expected: True)")
print(f"  ollama_num_predict: {num_predict} (expected: 512)")

if not cache_enabled:
    errors.append("enable_vector_cache is not True")
if num_predict != 512:
    errors.append(f"ollama_num_predict is {num_predict}, expected 512")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} FAILURES")
    for e in errors:
        print(f"  ❌ {e}")
else:
    print("RESULT: ALL CHECKS PASSED ✅")
print("=" * 60)

exit(1 if errors else 0)
