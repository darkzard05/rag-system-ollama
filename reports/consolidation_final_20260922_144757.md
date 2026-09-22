# Phase 4B Final Gates + Integrity Report

**Generated:** 2026-09-22 14:47:57 KST
**Branch:** main (working tree, uncommitted — 커밋 금지 준수)
**Baseline:** `reports/baseline_consolidation_20260922_113753.md` (101파일 / 23,091줄 @ a0778fc)
**Final:** 90파일 / 23,171줄 (+80줄, -11파일 net)

---

## 1. Shim 사용검사 → 삭제 (0건만 삭제, 잔류는 사유 기록)

검사 범위: src + tests + scripts 전수 Grep (import문 + patch 문자열 + docstring 제외 실사용).

### 1.1 삭제 (12파일, 실사용 0건 확인)

| # | 삭제 파일 | 구경로 |正규 홈 | 사용검사 결과 |
|---|----------|--------|--------|--------------|
| 1 | `src/common/retry.py` | `common.retry` | `common.resilience` | 0건 (실import/patch 없음) |
| 2 | `src/common/circuit_breaker.py` | `common.circuit_breaker` | `common.resilience` | 0건 |
| 3 | `src/common/_utils_text.py` | `common._utils_text` | `common.text_utils` | 0건 |
| 4 | `src/common/_utils_hash.py` | `common._utils_hash` | `common.utils` | 0건 |
| 5 | `src/common/_utils_cache.py` | `common._utils_cache` | `common.utils` | 0건 |
| 6 | `src/common/_utils_async.py` | `common._utils_async` | `common.async_worker` | 0건 |
| 7 | `src/common/_utils_annotations.py` | `common._utils_annotations` | `common.utils` | 0건 |
| 8 | `src/services/optimization/_backends.py` | `services.optimization._backends` | `services.optimization.backends` | 0건 (docstring 언급만) |
| 9 | `src/core/resource_pools.py` | `core.resource_pools` | `core.resource_manager` | 0건 (docstring 언급만) |
| 10 | `src/core/graph/_retrieve.py` | `core.graph._retrieve` | `core.graph._retrieval` | 0건 (sys.modules alias, 참조 없음) |
| 11 | `src/core/graph/_graph_cache.py` | `core.graph._graph_cache` | `core.graph._graph_core` | 0건 |
| 12 | `src/core/graph/_graph_internals.py` | `core.graph._graph_internals` | `core.graph._graph_core` | 0건 |

- 삭제 수단: `Remove-Item` (Bash). 해당 경로를 import하는 tests 없음 → tests 갱신 불필요 (MUST DO 조건 충족).
- `src/common/__init__.py`, `src/core/graph/__init__.py`(empty), `src/services/optimization/__init__.py`는 삭제 shim을 re-export하지 않음 확인 → 파급 없음.

### 1.2 잔류 (실사용 있음 → 유지)

| 잔류 shim | 사용 증거 | 유지 사유 |
|-----------|----------|----------|
| `src/core/embedding_memo.py` | `src/core/_loaders.py:131` lazy import + `tests/unit/conftest.py:55` + `test_embedding_memo.py:18` | canonical `core.model_loader` live-read shim, 실사용 중 |
| `src/cache/engine_cache.py` | `tests/test_loop_independence.py:6`, `test_engine_cache_reuse.py:10/142/167`, `test_phantom_state_rollback.py:15` | 로그 네임스페이스 보존 + 테스트 3파일 사용 |
| `src/core/graph/_grade.py` | `test_grade_reduction`, `test_structured_nodes`, `test_optimization_logic`, `test_graph_grading` 등 `patch("core.graph._grade.*")` 다수 | deep-patch 타깃 (sys.modules alias) |
| `src/core/graph/_verify.py` | `test_graph_verify_failopen`, `test_rag_state_propagation` (`core.graph._verify.*`) | 동상 |
| `src/core/graph/_preprocess.py` | `test_query_cache*.py`, `test_rag_state_propagation` (`core.graph._preprocess.*`) | 동상 |
| `src/core/graph/_generate.py` | canonical (실로직) + `test_*` 다수 patch | canonical이므로 삭제 대상 아님 |
| `src/core/graph/_speculative_gen.py` | canonical (실로직) + `test_speculative_generate` | 동상 |
| `src/api/streaming_handler.py` + `stream_events.py` + `stream_buffers.py` + `stream_state_machine.py` (4종) | `test_token_estimate`, `test_stream_ttft`, `test_streaming_response`, `test_streaming_*` 등 15+ 테스트 + `patch.object(api.streaming_handler, ...)` 시맨틱 | Phase 2C pure re-export, 테스트 계약 유지 |
| `src/ui/components/streaming.py` (facade, 실로직 포함) | `src/ui/components/chat.py:28/545` + 15+ 테스트 | facade + `consume_stream_into_message` 소유자 |
| `src/ui/components/streaming_state.py` / `streaming_runtime.py` / `streaming_extractors.py` (3종) | `test_stream_cancel` (runtime 4곳), `test_streaming_contract`/`test_error_recovery` (state), `test_main_background_tasks`, `test_generate_status_placeholder` | `patch.object(..._mod, ...)` 타깃 유지 |

---

## 2. CI FULL 게이트 (순서대로, 출력 기록)

| # | 게이트 | 결과 | 출력 |
|---|--------|------|------|
| 1 | `ruff check .` | ✅ PASS | `All checks passed!` |
| 2 | `ruff format --check .` | ✅ PASS | `91 files already formatted` (baseline 102 → Wave 병합+삭제 12 반영) |
| 3 | `mypy src` (PYTHONPATH=src) | ✅ PASS | `Success: no issues found in 90 source files` |
| 4 | `bandit -r src/ -ll` | ✅ PASS | `No issues identified` (17,810 LOC 스캔, Low 11のみ threshold 이하) |
| 5 | `pip-audit -r requirements.txt --skip-editable` | ✅ PASS | `No known vulnerabilities found` |
| 6a | `python scripts/maintenance/run_ci_checks.py` | ⚠️ 조건부 (아래) | import-sweep OK; integration 스위트 대부분 PASS, **1개 실패** + unit 페이즈 타임아웃(600s 초과, 별도 직접 실행으로 보완) |
| 6b | unit 직접 실행 `pytest tests/unit -q --cov=src --cov-fail-under=55` | ⚠️ 커버리지 PASS / 8 failed | `888 passed, 8 failed, 78.13% (gate 55% 상회)` |
| 7 | `python scripts/maintenance/verify_integrity.py` | ⏭️ SKIP-기록 | 내부적으로 run_ci_checks(600s) 재실행 구조라 단일 호출 300s 타임아웃; 구성 게이트를 개별 실행으로 대체 (ruff/mypy/bandit/pip-audit/unit-coverage/integration 단건 재현). Ollama E2E는 스크립트 내 주석 비활성화 상태. |

### 실패 상세 (전부 Wave1-3 기존 회귀, Phase 4B 삭제와 무관 — 삭제 12파일을 import하는 코드 0건)

Unit 8 failed:
- `test_cengine_parallel_fallback.py::test_cengine_fallback_parallel_matches_sequential` — `patch("core.document_processor.open_pdf_document")` 타깃 부재 (AttributeError). Wave 문서처리 병합 잔재.
- `test_coord_cache_eviction_batch.py::test_batch_eviction_...` — 1건.
- `test_generate_status_placeholder.py` — 2건 (status placeholder 계약).
- `test_main_background_tasks.py::test_stream_chunks_times_out_on_hanging_rag_stream` — 1건.
- `test_stream_cancel.py` — 3건 (timeout 시맨틱).

Integration:
- `tests/integration/test_chat_expander_verify_b.py::test_streamed_answer_renders_thought_expander_and_reenables_input` — 2회 연속 FAIL (재현 확인). `assert "Answer details" in expanders` vs 실제 `['Settings']`. UI 렌더 타이밍/라벨 회귀. 삭제 shim과 무관 (해당 테스트는 `RAGSystem.astream` mock + `src/main.py` AppTest만 사용).
- 그 외 integration 스위트 PASS: rag_integration 13, streaming_response 34, cache_security 34, caching_system 28+1xfail, api_auth_login 6, api_pdf_serving 2, global_exception 1, ownership 3, pdf_retention 5, stream_error_isolation 1, api_endpoints 5+1xfail, thread_safety 20.

---

## 3. 무동작변경 확인

- `git diff --stat -- src/api/`: 6파일, +67 / -880 — `streaming_handler/stream_events/stream_buffers/stream_state_machine`의 Phase 2C shim화(삭제가 아닌 re-export 전환) + `routes_chat.py` 6줄·`sse_helpers.py` 2줄 import 경로 조정. 라우트 9개 전수 Grep 확인 → **전부 보존** (`/api/v1/login, logout, health, admin/stats, query, stream_query, session/{id}, upload, pdf/{hash}`).
- `git diff -- config.yml`: **empty** (top keys 7종 + `rag.prompts` 7종 `analysis_protocol/generate/grade/grading/prompt_templates/token_estimation_proxy/verify` 불변).
- prompts diff: config.yml empty이므로 **empty** (외부 prompt 파일 없음).
- 커버리지: **78.13% ≥ 55%** ✅ (`TOTAL 10167 stmts, 2224 miss`).

---

## 4. 위상별 파일수 델타 (baseline → final)

| 위상 | 파일수 | 비고 |
|------|--------|------|
| Baseline (a0778fc) | **101** (23,091줄) | `reports/baseline_consolidation_20260922_113753.md` |
| Wave1-3 (작업트리, Task9 포함) | 102* 추정 | canonical 8종 추가 (`stream_pipeline, resilience, _grade_verify, _graph_core, _retrieval, backends, streaming_core, monitoring/notification_system`), `_bg_indexing.py`·`infra/notification_system.py`·`_backends_*` 5종 삭제, tests import 갱신 |
| Phase 4B (본 작업) | **90** (23,171줄, +80) | 0건 shim 12종 삭제 (위 §1.1) |
| **Net 델타** | **101 → 90 (−11)** | 산식: 101 −1(_bg) −1(infra) −5(_backends_*분할) −12(본작업) +8(canonical) = 90 ✅ (실측 `mypy 90 files`, `Get-ChildItem 90`, python rglob 90 일치) |

*Wave 중간 스냅샷 리포트 부재 — git working-tree diff로 역산. LOC +80은 canonical 병합 시 주석·타입 보강분.

---

## 5. Test → Src 맵 (갱신, 29 integration + unit 커버리지)

| Test File | Mapped Src (최종 경로) |
|-----------|----------------------|
| `test_rag_integration.py` (13) | `core/rag_core.py`, `core/graph/graph_builder.py`, `core/pipeline_builder.py` |
| `test_streaming_response.py` (34) | `api/stream_pipeline.py` (via `streaming_handler` shim), `ui/components/streaming_core.py` (via `streaming` facade) |
| `test_streamlit_ui_lifecycle.py` | `main.py`, `ui/ui.py`, `ui/components/chat.py` |
| `test_api_auth_login.py` (6) | `api/api_server.py`, `security/auth_system.py` |
| `test_api_pdf_serving.py` (2) | `api/routes_docs.py`, `core/document_processor.py` |
| `test_api_endpoints.py` (5+1xfail) | `api/api_server.py`, `api/routes_chat.py`, `api/routes_docs.py` |
| `test_caching_system.py` (28+1xfail) | `services/optimization/caching_optimizer.py` (+`backends.py`), `cache/vector_cache.py` |
| `test_pipeline_build_once.py` | `core/pipeline_builder.py`, `core/graph/graph_builder.py` |
| `test_global_exception_handler.py` (1) | `api/api_server.py`, `common/exceptions.py` |
| `test_ownership_hardening.py` (3) | `security/auth_system.py`, `security/cache_security.py` |
| `test_pdf_library_retention.py` (5) | `core/document_processor.py`, `cache/coord_cache.py` |
| `test_stream_error_isolation.py` (1) | `api/stream_pipeline.py`, `ui/components/streaming_core.py` |
| `test_consecutive_queries.py` | `core/rag_core.py`, `core/graph/_generate.py` |
| `test_chat_expander_verify_a/b.py` | `ui/components/chat.py`, `ui/components/chat_build.py` — ⚠️ b 1건 FAIL (위 §2) |
| `test_defects_verification.py` | `core/document_hydrator.py`, `cache/coord_cache.py` |
| `test_metadata_propagation.py` | `core/document_processor.py`, `core/chunking.py` |
| `test_dynamic_model_loading.py` | `core/model_loader.py`, `core/custom_ollama.py` |
| `test_performance_monitor.py` | `services/monitoring/performance_monitor.py` |
| `test_session_management.py` | `core/session/manager.py` |
| `test_upload_flow.py` | `api/routes_docs.py`, `core/document_processor.py` |
| `test_stream_ttft.py` | `api/stream_pipeline.py`, `ui/components/streaming_core.py` |
| `test_thread_safety.py` (20) | `core/session/manager.py`, `services/monitoring/performance_monitor.py` |
| `test_ui_persistence_real.py` | `ui/ui.py`, `ui/components/viewer.py` |
| `test_input_boundaries.py` | `core/semantic_chunker.py`, `core/document_processor.py` |
| `test_rag_state_propagation.py` | `core/graph/graph_builder.py` (+alias `_grade/_verify/_preprocess`), `core/session/manager.py` |
| `test_resource_optimization.py` | `core/resource_manager.py` (pools 흡수), `services/optimization/caching_optimizer.py` |
| `test_chat_metrics_rendering.py` | `ui/components/chat.py`, `ui/components/chat_build.py` |
| `test_streaming_status_empty_body.py` | `api/stream_pipeline.py`, `ui/components/streaming_core.py` |
| `test_auth_bootstrap.py` | `security/auth_system.py`, `api/api_server.py` |
| unit `test_common_retry` / `test_circuit_breaker_transitions` | `common/resilience.py` (구 shim 삭제 후 canonical 직접) |
| unit `test_backends_spike` / `test_object_cache` / `test_caching_optimizer` | `services/optimization/backends.py` |
| unit `test_engine_cache_reuse` 등 3파일 | `cache/engine_cache.py` shim → `cache/vector_cache.py` (유지) |
| unit `test_embedding_memo` / `conftest` | `core/embedding_memo.py` shim → `core/model_loader.py` (유지) |

Unit 총계: **888 passed, 8 failed** (실패는 §2 목록, Wave 기존 회귀). Integration 디렉토리 파일수: 31.

---

## 6. 잔류 shim 최종 목록 (삭제 금지)

`common`: 없음 (7종 전부 삭제). `services.optimization._backends`: 삭제. `core.resource_pools`: 삭제. `core.graph`: `_grade, _verify, _preprocess` (alias 유지) + canonical `_generate, _speculative_gen, _grade_verify, _retrieval, _graph_core, _glue`. 삭제: `_retrieve, _graph_cache, _graph_internals`. `api` 4종 전부 유지. `ui` facade+3종 전부 유지. `core.embedding_memo`, `cache.engine_cache` 유지.

## 7. 결론

- MUST DO 1 (shim): 12개 0건 삭제 완료, 잔류 사유 기록 ✅
- MUST DO 2 (게이트): ruff/mypy/bandit/pip-audit **그린** ✅, coverage **78.13%** ✅, run_ci_checks **조건부** (integration 1 + unit 8, 전부 Wave 기존 회귀 — Phase 4B 무관 근거 첨부) ⚠️
- MUST DO 3 (무동작): api 이동-only, config/prompts empty, routes 9 보존 ✅
- MUST NOT DO: src 신규병합·개명·게이트완화·커밋 **없음** ✅ (변경은 shim 12삭제のみ)
- 후속 권장: `test_cengine_parallel_fallback` patch 타깃 복구, `stream_cancel` timeout 시맨틱, `chat_expander_verify_b` expander 라벨 — 모두 Wave 스코프이며 본 Phase 밖.
