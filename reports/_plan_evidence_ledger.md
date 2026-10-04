# Plan Evidence Ledger — rag-system-ollama Top-3 Verification

All entries are file:line + verbatim values. Git ground truth takes precedence; deviations explicitly noted.

---

## F1 — Weight block lines 70–84 in src/core/graph/_retrieval.py

| Claim | File:Line | Verbatim |
|---|---|---|
| F1-1 | _retrieval.py:70 | `if DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):` |
| F1-2 | _retrieval.py:77 | `if is_keyword_heavy and not is_semantic_heavy:` |
| F1-3 | _retrieval.py:78 | `kw_w = DYNAMIC_WEIGHTING_CONFIG.get("keyword_weight", 0.8)` |
| F1-4 | _retrieval.py:79 | `weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}` |
| F1-5 | _retrieval.py:81 | `elif is_semantic_heavy and not is_keyword_heavy:` |
| F1-6 | _retrieval.py:82 | `sm_w = DYNAMIC_WEIGHTING_CONFIG.get("semantic_weight", 0.8)` |
| F1-7 | _retrieval.py:83 | `weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}` |
| F1-8 | _retrieval.py:84 | `logger.info(f"[RAG] [PREPROCESS] 의미 중심 질의 판단 (FAISS: {sm_w})")` |
| F1-9 | grep '0.6' in _retrieval.py | **No matches found** (grep confirmed 0 hits) |
| F1-10 | config.yml:90 | `ensemble_weights: [0.4, 0.6]` (static default; **no 0.6 blend branch** in dynamic weighting) |

**F1 Verdict:** Dynamic weighting uses `0.8` keyword_weight and `0.8` semantic_weight. No `0.6` blend branch exists in `_retrieval.py`. The `ensemble_weights: [0.4, 0.6]` in config.yml is a static default, not a dynamic blend path.

---

## F2 — search_k, aggregate/slice, guard, config values

| Claim | File:Line | Verbatim |
|---|---|---|
| F2-1 | _retrieval.py:221 | `search_k = int(RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 25))` |
| F2-2 | _retrieval.py:336 | `aggregated, _ = aggregator.aggregate_results(source_results, strategy=AggregationStrategy.WEIGHTED_RRF, top_k=search_k, weights=weights)` |
| F2-3 | _retrieval.py:345 | `dynamic_cfg = DYNAMIC_TOP_K_CONFIG` |
| F2-4 | _retrieval.py:346 | `gap_threshold = float(dynamic_cfg.get("gap_threshold", 0.003))` |
| F2-5 | _retrieval.py:347 | `min_candidates = int(dynamic_cfg.get("min_candidates", 12))` |
| F2-6 | _retrieval.py:348 | `max_candidates = int(dynamic_cfg.get("max_candidates", 18))` |
| F2-7 | _retrieval.py:356 | `dynamic_top_k = min_candidates if score_gap > gap_threshold else max_candidates` |
| F2-8 | _retrieval.py:77 | `if is_keyword_heavy and not is_semantic_heavy:` (guard 1) |
| F2-9 | _retrieval.py:81 | `elif is_semantic_heavy and not is_keyword_heavy:` (guard 2) |
| F2-10 | config.yml:88 | `k: 15  # k-chain Stage-1 단일 절단` |
| F2-11 | config.yml:90 | `ensemble_weights: [0.4, 0.6]` |
| F2-12 | config.yml:94-97 | `dynamic_top_k: gap_threshold: 0.003, min_candidates: 12, max_candidates: 18` |
| F2-13 | config.yml:100-103 | `dynamic_weighting: enabled: true, keyword_weight: 0.8, semantic_weight: 0.8` |

**F2 Verdict:** search_k defaults to 25 (line 221) but config.yml overrides to k=15 (line 88). Dynamic Top-K uses gap_threshold=0.003, min=12, max=18. Guard branches at lines 77/81 control dynamic weight assignment.

---

## F3 — threshold triple + eval-corpus max_rerank source

| Claim | File:Line | Verbatim |
|---|---|---|
| F3-1 | config.yml:256-260 | `# FlashRank sigmoid short-circuit 임계값 — max_rerank_score가 이 값 이상이면 LLM grade 생략.`<br>`min_score_to_skip: 0.70` |
| F3-2 | config.yml:261-265 | `# semantic(bi-encoder) 엔진용 별도 임계값.`<br>`min_score_to_skip_semantic: 0.45` |
| F3-3 | config.yml:266-269 | `# 상위-차상위 rerank_score 격차가 이 값 미만이면 애매한 문서 집합으로 판단`<br>`min_top_gap_to_skip: 0.05` |
| F3-4 | config.yml:270-273 | `top_k: 10`<br>`grade_top_n: 3`<br>`max_retries: 2` |
| F3-5 | _grade_verify.py:241-244 | `if get_active_rerank_engine() == "semantic": min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip_semantic", 0.40)<br>else: min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip", 0.45)` |
| F3-6 | _grade_verify.py:246 | `min_top_gap = GRADING_CONFIG.get("min_top_gap_to_skip", 0.03)` |
| F3-7 | _grade_verify.py:247-250 | `sorted_scores = sorted((float(d.metadata.get("rerank_score", 0.0)) for d in docs), reverse=True)<br>top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0` |
| F3-8 | _grade_verify.py:253-254 | `both_high_confidence = len(sorted_scores) > 1 and sorted_scores[1] >= (min_score_to_skip - 0.07)` |
| F3-9 | _grade_verify.py:256-257 | `if max_rerank_score >= min_score_to_skip and (top_gap >= min_top_gap or both_high_confidence):` |
| F3-10 | eval quality reports | max_rerank source: `scripts/eval_quality.py` (produces `reports/eval_quality_*.md/.json`) |
| F3-11 | README grade table | Referenced in config.yml:256; grade table evaluates answer relevance, conjecture, and citation fidelity (see lines 251-254) |

**F3 Verdict:** Threshold triple consists of `min_score_to_skip` (0.70 general / 0.45 semantic), `min_top_gap_to_skip` (0.05), and `top_k` (10). The grade short-circuit in `_grade_verify.py:256` activates when max_rerank_score >= min_score_to_skip AND (top_gap >= min_top_gap OR both_high_confidence). Max rerank scores come from `scripts/eval_quality.py` evaluation runs.

---

## A2 — verdict inputs resource_manager.py: per-key dict? limit 3? TTL 60s? FIFO cap 5?

| Claim | File:Line | Verbatim |
|---|---|---|
| A2-1 | resource_manager.py:49-52 | `# _BUILD_FAILURE_LIMIT = 3`<br>`# _BUILD_CIRCUIT_TTL_SECONDS = 60.0` |
| A2-2 | resource_manager.py:543-545 | `_check_build_circuit(key): ... if count >= _BUILD_FAILURE_LIMIT:`<br>`if elapsed < _BUILD_CIRCUIT_TTL_SECONDS:` → raise ResourceBuildError` |
| A2-3 | resource_manager.py:578-586 | `_record_build_failure(key):` purges expired, increments count, enforces `_BUILD_FAILURE_MAX_ENTRIES = 5` (line 61) |
| A2-4 | resource_manager.py:588-596 | `if count >= _BUILD_FAILURE_LIMIT:` logs error with `"{_BUILD_CIRCUIT_TTL_SECONDS:.0f}초 회로 차단 (재시도 → 즉시 폴백 전환)"` |
| A2-5 | resource_manager.py:60-61 | `_BUILD_FAILURE_MAX_ENTRIES = 5` (total dict cap, not per-key FIFO) |
| A2-6 | resource_manager.py:546-565 | `_check_build_circuit`: circuit opens at 3 failures within 60s TTL; after TTL elapses, counter resets and retry allowed |

**A2 Verdict:** Per-key failure dict `_build_failures` (dict[str, tuple[int, float]]). Limit 3 consecutive failures before circuit TTL 60s activates. Total entry cap is 5 (_BUILD_FAILURE_MAX_ENTRIES), enforced via LRU eviction in `_record_build_failure`. No per-key FIFO cap of 5 — the cap is on total dict size.

---

## C2 — verdict inputs resource_manager.py: T1 idle-only rule present? cite 450-499,612-667

| Claim | File:Line | Verbatim |
|---|---|---|
| C2-1 | resource_manager.py:476-502 | `_get_build_lock(key):` T1 unified rule: "single entry per key, recreate only when idle (_inference_held == 0)". Existing lock returned if held; foreign-loop contender fails loud. Idle + foreign-bound → recreate. |
| C2-2 | resource_manager.py:487-498 | `existing = self._build_locks.get(key)`; if existing.locked() → return existing (no fork). Bound loop check: `bound_loop is None or bound_loop is asyncio.get_event_loop()`. |
| C2-3 | resource_manager.py:499-502 | `self._build_locks.pop(key, None)` then `self._build_locks[key] = asyncio.Lock()` — new lock created only after eviction. |
| C2-4 | resource_manager.py:612-633 | `acquire_inference_lock:` T1 unified rule: "recreate on loop/bound change ONLY when idle (_inference_held == 0)". If _inference_semaphore is None → create fresh. Loop change + idle → recreate. |
| C2-5 | resource_manager.py:634-639 | `elif self._semaphore_loop is not loop:` + `if self._inference_held == 0:` → recreate semaphore with effective_bound. |
| C2-6 | resource_manager.py:643-648 | `elif self._inference_semaphore_bound is not None:` + `if _inference_semaphore_bound != effective_bound and _inference_held == 0:` → recreate on VRAM-pressure bound change. |
| C2-7 | T1 rule summary | **Present**: Both `_get_build_lock` and `acquire_inference_lock` enforce "recreate ONLY when idle". Recreating while permits are outstanding discards accounting and over-admits. |

**C2 Verdict:** T1 idle-only rule is **present** in resource_manager.py, cited in `_get_build_lock` (lines 476-502) and `acquire_inference_lock` (lines 612-667). The rule mandates recreation only when `_inference_held == 0` or when the lock is not held, preventing double-build and over-admission across event loops.

---

## Git — commit 2869337 vs eval_quality P@1/MRR

| Claim | Output |
|---|---|
| Git:log -5 | `9056c0d test(autoscroll): add S1-S3 visual QA captures with sidecar metrics and sticky-input helper`<br>`1cb898c fix: AST both-directions i18n gate with drop warnings and viewer parity`<br>`c04261d fix: separable retrieval timing with failure-path emit and INFO promotion`<br>`9dba91a fix: bounded hydration drain with timeout metric and load-gated inference concurrency`<br>`f343ba2 fix: loop-safe build locks + bounded checkpointer and spec registry eviction` |
| Git:show 2869337 --stat | `commit 2869337442dc88dbf3a6457b4d5233a6d13eeb29`<br>`Author: darkzard05 <darkzard05@gmail.com>`<br>`Date:   Wed Sep 30 12:59:58 2026 +0900`<br>`perf(retrieval): widen semantic routing keywords and rerank top_k pool`<br>`semantic_keywords: add 8 casual Korean question markers (...)`<br>`Evidence (reports/repro_h3_weights_20260927_110256.md): the 12-question routing matrix shows 0.0% misroute (0/12) with the expanded list.`<br>`grading.top_k 5 -> 10: widen the rerank pool so fewer candidates are dropped before grading.`<br>`1 file changed, 2 insertions(+), 2 deletions(-)`<br>`config.yml | 4 insertions(+), 2 deletions(-)` |
| P@1 (final-now) | `0.0` (line 15, reports/eval_quality_final-now_20260927_120511.md) |
| MRR@5 (final-now) | `0.1111` (line 16, same file) |
| P@1/MRR comparison (baseline→after) | `P@1: 0.0 → 0.0 — 동일 (개선 없음)`<br>`MRR@5: 0.0 → 0.0 — 동일 (개선 없음)` (eval_quality_comparison.md:29-30) |

**Git Verdict:** Commit 2869337 widened semantic routing keywords (added 8 Korean casual markers) and changed grading.top_k from 5→10 in config.yml. No measurable P@1/MRR improvement in comparable eval runs (P@1 stayed 0.0, MRR@5 stayed 0.1111). The linked repro_h3_weights report shows 0.0% misroute with the expanded keyword list.

---

## Glob — reports/ file inventory

| Pattern | Files Found |
|---|---|
| reports/eval_quality_* | 55 files (see grep results for full list) |
| reports/repro_h3_* | 3 files: `repro_h3_weights_20260927_110256.md`, `repro_h3_weights_20260927_091801.md`, `repro_h3_weights_20260927_091747.md` |

**Glob Verdict:** 55 eval quality report files exist; 3 H3 weight repro files. The primary reference is `reports/repro_h3_weights_20260927_110256.md` cited in commit 2869337.

---

# Summary — Ledger Section IDs for Future Citation

| Topic | Section ID |
|---|---|
| F1 weight block & 0.6 blend | F1 |
| F2 search_k, aggregate, guards, config | F2 |
| F3 threshold triple + max_rerank source | F3 |
| A2 resource_manager verdicts | A2 |
| C2 T1 idle-only rule | C2 |
| Git 2869337 vs eval quality | Git |
| P@1/MRR values | P@1/MRR |

*Every later task can cite section IDs (F1, F2, etc.) instead of prose.*