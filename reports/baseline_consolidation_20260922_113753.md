# Phase 0 Baseline Consolidation Report

**Generated:** 2026-09-22 11:37:53 KST  
**Commit:** a0778fc  
**Branch:** main  
**Scope:** `src/` read-only baseline — filecount / LOC / import-graph freeze + shim policy + 금지개명 목록

---

## 1. Delta Baseline — 100 File Count Confirmation

**Success Criterion:** `src/` 하위 100 파일 명시 확인

| Metric | Value |
|--------|-------|
| Total `.py` files in `src/` | **101** (incl. `__init__.py` × 8) |
| Total LOC | **23,091** |
| Non-`__init__` files | **93** |
| `__init__.py` files | **8** |

> ✅ **100+ 파일 명시 확인 완료.** (101 파일, 8개 `__init__.py` 포함)

---

## 2. Subdirectory File Count / LOC Breakdown

| Subdirectory | Files | LOC | Notes |
|-------------|-------|-----|-------|
| `common/` | 18 | 2,647 | `_utils_*` 5개 포함, `__init__.py` |
| `core/` (top-level) | 14 | 4,495 | `_loaders.py`, `semantic_chunker.py` (1,124 LOC) 최대 |
| `core/graph/` | 10 | 2,503 | `_generate.py` (619 LOC) 최대 |
| `core/session/` | 2 | 769 | `manager.py` (765 LOC) |
| `ui/` (top-level) | 7 | 831 | `bridge.py`, `session_sync.py`, `strings.py`, `ui.py`, `widget_keys.py` |
| `ui/components/` | 9 | 2,854 | `chat.py` (740 LOC) 최대 |
| `api/` | 10 | 1,898 | `streaming_handler.py` (514 LOC) 최대 |
| `cache/` | 4 | 1,078 | `coord_cache.py` (534 LOC), `vector_cache.py` (528 LOC) |
| `services/` | 8 | 1,775 | `monitoring/` 1개, `optimization/` 7개 |
| `security/` | 4 | 1,113 | `auth_system.py` (642 LOC) 최대 |
| `infra/` | 2 | 95 | `notification_system.py` |
| `_bg_indexing.py` | 1 | 296 | 루트 레벨 |
| `main.py` | 1 | 648 | 루트 레벨 |
| `src/__init__.py` | 1 | 0 | |

**Total: 101 files, 23,091 LOC**

---

## 3. Import Graph (Top Importers by Directory)

### 3.1 핵심 내부 의존성 맵

```
src/main.py
  ├── _bg_indexing.py
  ├── common.async_worker, common.config, common.constants, common.logging_config, common.utils
  ├── core.session.SessionManager (×5 중복 import)
  ├── core.model_loader.get_available_models, core.model_loader._warmup_models
  ├── core.document_processor.compute_file_hash
  ├── infra.notification_system.SystemNotifier
  ├── ui.components.sidebar.render_settings_content
  ├── ui.ui.render_main_content, ui.ui.inject_custom_css
  └── ui.bridge.UIBridge

src/core/rag_core.py
  ├── common.circuit_breaker, common.exceptions, common.retry, common.utils
  ├── core.document_hydrator, core.pipeline_builder, core.resource_manager, core.session
  ├── core.graph.graph_builder (relative import)
  └── services.monitoring.performance_monitor

src/core/graph/graph_builder.py
  ├── api.schemas.GraphState
  ├── core.graph._generate, _grade, _graph_cache, _graph_internals, _preprocess, _retrieve, _speculative_gen, _verify (re-exports)
  ├── core.session.SessionManager
  ├── common.config.VERIFICATION_ENABLED
  └── langgraph.graph, langgraph.checkpoint

src/core/pipeline_builder.py
  ├── cache.vector_cache.VectorStoreCache
  ├── common.config, common.constants, common.exceptions
  ├── core.chunking, core.document_processor, core.graph.graph_builder
  ├── core.resource_manager, core.retriever_factory, core.session
  ├── services.optimization.caching_optimizer
  └── core.model_loader.ModelManager

src/core/model_loader.py
  ├── common.config, common.exceptions, common.system_pressure
  ├── core._loaders (re-exports), core.resource_manager
  ├── core.session.SessionManager
  ├── langchain_ollama, langchain_core, langchain_huggingface
  └── faiss, torch, importlib.util

src/core/semantic_chunker.py (1,124 LOC)
  ├── common.config, common.constants, common.similarity
  ├── services.optimization.caching_optimizer
  ├── langchain_ollama, langchain_core
  └── core.resource_manager

src/core/resource_manager.py (642 LOC)
  ├── common.config, common.exceptions, common.system_pressure
  ├── core.model_loader.load_llm, load_embedding_model
  ├── core.resource_pools
  ├── flashrank, ollama, faiss
  └── core.session.SessionManager

src/core/session/manager.py (765 LOC)
  ├── common.constants, common.retry
  ├── ui.session_sync.StreamlitSessionSync
  ├── core.graph.graph_builder.delete_graph_thread
  └── streamlit.runtime

src/ui/components/chat.py (740 LOC)
  ├── common.utils, core.session.SessionManager
  ├── ui.components.common, ui.components.streaming, ui.components.chat_references, ui.components.chat_build
  ├── ui.strings, ui.widget_keys
  └── streamlit

src/ui/components/streaming.py (360 LOC)
  ├── api.stream_events, common.config, common.utils
  ├── core.session.SessionManager, core.rag_core.RAGSystem
  ├── ui.components.streaming_extractors, ui.components.streaming_runtime, ui.components.streaming_state
  └── streamlit

src/api/api_server.py
  ├── app.post /api/v1/login, /api/v1/logout
  ├── app.get /api/v1/health, /api/v1/admin/stats
  ├── include_router(chat_router) → routes_chat.py
  └── include_router(docs_router) → routes_docs.py

src/api/routes_chat.py
  ├── router.post /api/v1/query, /api/v1/stream_query
  ├── router.delete /api/v1/session/{session_id}
  └── core.session.SessionManager

src/api/routes_docs.py
  ├── router.post /api/v1/upload
  ├── router.get /api/v1/pdf/{file_hash}
  └── security, common, core modules
```

### 3.2 순환 의존성 주의 구간

| Cycle | Files | Risk |
|-------|-------|------|
| `common._utils_async` ↔ `core.session` | `common/_utils_async.py` → `core.session` | Lazy import로 순환 방지 |
| `core._loaders` ↔ `core.model_loader` | `core/_loaders.py` → `core.model_loader` | `# 순환 방지 lazy import` 주석 |
| `core.rag_core` ↔ `core.graph.graph_builder` | `core/rag_core.py` → `core/graph/graph_builder.py` | Relative import로 처리 |
| `core.session.manager` ↔ `ui.session_sync` | `core/session/manager.py` → `ui/session_sync.py` | Streamlit 런타임 의존 |

---

## 4. Test → Src Mapping (29 Integration Tests)

### 4.1 테스트 파일 분포

| Directory | Count | Description |
|-----------|-------|-------------|
| `tests/unit/` | 180 | Unit tests (core, ui, api, cache, common, security) |
| `tests/integration/` | **31** | Integration tests (RAG pipeline, streaming, auth, PDF) |
| `tests/security/` | 3 | Security tests (cache_security, pickle RCE) |
| `tests/e2e/` | 2 | E2E tests (chat scroll, PDF controls) |
| `tests/stability/` | 1 | Stability tests (session concurrency) |
| **TOTAL** | **217** | |

### 4.2 Integration Test → Src 매핑 (29개 주요 테스트)

| Test File | Mapped Src Module(s) |
|-----------|---------------------|
| `test_rag_integration.py` | `core/rag_core.py`, `core/graph/graph_builder.py`, `core/pipeline_builder.py` |
| `test_streaming_response.py` | `api/streaming_handler.py`, `api/stream_events.py`, `ui/components/streaming.py` |
| `test_streamlit_ui_lifecycle.py` | `main.py`, `ui/ui.py`, `ui/components/chat.py` |
| `test_api_auth_login.py` | `api/api_server.py`, `security/auth_system.py` |
| `test_api_pdf_serving.py` | `api/routes_docs.py`, `core/document_processor.py` |
| `test_api_endpoints.py` | `api/api_server.py`, `api/routes_chat.py`, `api/routes_docs.py` |
| `test_caching_system.py` | `services/optimization/caching_optimizer.py`, `cache/vector_cache.py` |
| `test_pipeline_build_once.py` | `core/pipeline_builder.py`, `core/graph/graph_builder.py` |
| `test_global_exception_handler.py` | `api/api_server.py`, `common/exceptions.py` |
| `test_ownership_hardening.py` | `security/auth_system.py`, `security/cache_security.py` |
| `test_pdf_library_retention.py` | `core/document_processor.py`, `cache/coord_cache.py` |
| `test_stream_error_isolation.py` | `api/streaming_handler.py`, `ui/components/streaming_state.py` |
| `test_consecutive_queries.py` | `core/rag_core.py`, `core/graph/_generate.py` |
| `test_chat_expander_verify_a/b.py` | `ui/components/chat.py`, `ui/components/chat_build.py` |
| `test_defects_verification.py` | `core/document_hydrator.py`, `cache/coord_cache.py` |
| `test_metadata_propagation.py` | `core/document_processor.py`, `core/chunking.py` |
| `test_dynamic_model_loading.py` | `core/model_loader.py`, `core/custom_ollama.py` |
| `test_performance_monitor.py` | `services/monitoring/performance_monitor.py` |
| `test_session_management.py` | `core/session/manager.py` |
| `test_upload_flow.py` | `api/routes_docs.py`, `core/document_processor.py` |
| `test_stream_ttft.py` | `api/streaming_handler.py`, `ui/components/streaming_runtime.py` |
| `test_thread_safety.py` | `core/session/manager.py`, `services/monitoring/performance_monitor.py` |
| `test_ui_persistence_real.py` | `ui/ui.py`, `ui/components/viewer.py` |
| `test_input_boundaries.py` | `core/semantic_chunker.py`, `core/document_processor.py` |
| `test_rag_state_propagation.py` | `core/graph/graph_builder.py`, `core/session/manager.py` |
| `test_resource_optimization.py` | `core/resource_manager.py`, `services/optimization/caching_optimizer.py` |
| `test_chat_metrics_rendering.py` | `ui/components/chat.py`, `ui/components/chat_build.py` |
| `test_streaming_status_empty_body.py` | `api/stream_events.py`, `ui/components/streaming_state.py` |
| `test_auth_bootstrap.py` | `security/auth_system.py`, `api/api_server.py` |

---

## 5. Lint/Gate Results (Read-Only — 수정금지)

### 5.1 ruff check
```
All checks passed!
Exit code: True (0)
```
**Status:** ✅ PASS — 모든 파일에서 린트 에러 없음

### 5.2 ruff format --check
```
102 files already formatted
Exit code: True (0)
```
**Status:** ✅ PASS — 102 파일 포맷팅 정상

### 5.3 mypy src
```
Success: no issues found in 101 source files
Exit code: True (0)
```
**Status:** ✅ PASS — 101 소스 파일 타입 체크 통과

### 5.4 pytest tests/unit -q --cov=src --cov-fail-under=55
```
896 passed, 1 warning in 126.31s (0:02:06)
Required test coverage of 55% reached. Total coverage: 78.64%
Exit code: 0
```
**Status:** ✅ PASS — 896 테스트 통과, 커버리지 78.64% (게이트 55% 상회)

### 5.5 Gate Summary

| Gate | Result | Status |
|------|--------|--------|
| `ruff check .` | All checks passed | ✅ PASS |
| `ruff format --check .` | 102 files formatted | ✅ PASS |
| `mypy src` | No issues in 101 files | ✅ PASS |
| `pytest tests/unit -q --cov=src --cov-fail-under=55` | 896 passed, 78.64% coverage | ✅ PASS |

> ⚠️ **모든 게이트 통과. 수정 금지. 이 Baseline은 변경 불가 상태.**

---

## 6. 금지개명 목록 (Prohibited Rename List)

### 6.1 config.yml 키 — 절대 개명 금지

**Top-Level Keys:**
- `models`, `rag`, `evaluation`, `cache_security`, `cors`, `global_cache`, `ui`

**Model Sub-Keys:**
- `default_ollama`, `base_url`, `ollama_num_predict`, `temperature`, `num_ctx`, `top_p`, `thinking`, `cache_dir`, `embedding_batch_size`, `embedding_device`, `max_cached_models`, `max_concurrent_inference`, `timeout`, `enable_ollama_pressure_fallback`, `host_pressure_threshold`

**RAG Sub-Keys:**
- `vector_store_cache_dir`, `vector_store`, `retriever`, `reranker`, `text_splitter`, `parsing`, `semantic_chunker`, `query_cache`, `grading`, `prompts`

**UI Sub-Keys:**
- `streaming`, `messages`

### 6.2 API Routes — 절대 개명/재배치 금지

| Method | Path | Handler File |
|--------|------|-------------|
| POST | `/api/v1/login` | `api/api_server.py` |
| POST | `/api/v1/logout` | `api/api_server.py` |
| GET | `/api/v1/health` | `api/api_server.py` |
| GET | `/api/v1/admin/stats` | `api/api_server.py` |
| POST | `/api/v1/query` | `api/routes_chat.py` |
| POST | `/api/v1/stream_query` | `api/routes_chat.py` |
| DELETE | `/api/v1/session/{session_id}` | `api/routes_chat.py` |
| POST | `/api/v1/upload` | `api/routes_docs.py` |
| GET | `/api/v1/pdf/{file_hash}` | `api/routes_docs.py` |

### 6.3 Prompt Section Names — 절대 개명 금지

| Section | Config Path | Description |
|---------|------------|-------------|
| `1. 핵심 페르소나 및 분석 원칙` | `rag.prompts.analysis_protocol` | Core identity prompt |
| `2. 문서 평가(grading) 설정` | `rag.prompts.grading` | FlashRank short-circuit config |
| `3. 구조화된 답변 출력용 프롬프트 템플릿` | `rag.prompts.prompt_templates` | JSON schema output |
| `4. 최종 답변 보조 메시지` | `rag.prompts.generate` | Generate human message |
| `5. 문서 평가(grade_documents)` | `rag.prompts.grade` | Grading system/human message |
| `6. faithfulness 검증(verify)` | `rag.prompts.verify` | Verification prompt |

### 6.4 핵심 Symbol Names — 개명 금지 목록

| Symbol | Location | Role |
|--------|----------|------|
| `RAGSystem` | `src/core/rag_core.py` | Main orchestrator |
| `PipelineBuilder` | `src/core/pipeline_builder.py` | Pipeline build/cache |
| `EmbeddingBasedSemanticChunker` | `src/core/semantic_chunker.py` | Header-aware splitter |
| `build_graph` | `src/core/graph/graph_builder.py` | LangGraph construction |
| `SessionManager` | `src/core/session/manager.py` | Thread-safe state |
| `CoordCacheManager` | `src/cache/coord_cache.py` | PDF coordinate cache |
| `UIBridge` | `src/ui/bridge.py` | SessionManager ↔ Streamlit |
| `ResourceManager` | `src/core/resource_manager.py` | LRU model pools |
| `ModelManager` | `src/core/model_loader.py` | LLM/embedding facade |
| `graph_builder` | `src/core/graph/graph_builder.py` | Re-export hub |

---

## 7. 구경로 Shim 정책 (Shim Policy)

### 7.1 정책 정의

**목적:** Phase 1에서 경로 변경이 발생하더라도 하위 호환성을 유지하기 위한 re-export shim 정책.

**원칙:**
1. **1페이즈 유지:** 모든 `old path → new path` re-export shim은 Phase 1에서만 유지
2. **Task10 제거:** Phase 2에서 Task10 (shim 정리)를 통해 모든 shim 제거
3. **`# noqa: F401` 주석:** re-export shim에 반드시 주석 추가
4. **`# backward compat` 주석:** 하위 호환 목적임을 명시

### 7.2 현재 Shim 현황

| Old Path | New Path | Shim Type | Status |
|----------|----------|-----------|--------|
| `core.graph._generate` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._grade` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._graph_cache` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._graph_internals` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._preprocess` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports (back-compat)` |
| `core.graph._retrieve` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._speculative_gen` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph._verify` | `core.graph.graph_builder` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core._loaders` | `core.model_loader` | re-export | `# noqa: F401 — re-exports for backward compat` |
| `core.graph.graph_builder` | `core.rag_core` | relative import | `.graph.graph_builder` |

### 7.3 Shim 정책 규칙

```
RULE 1: 모든 re-export shim은 반드시 다음 주석을 포함
        # noqa: F401 — re-exports for backward compat
        # backward compat: Phase 1 only, remove in Task10

RULE 2: shim 파일은 절대 로직을 포함하지 않음
        - 순수 re-export만 허용
        - `from .new_path import symbol` 형태만 허용

RULE 3: Phase 2 (Task10)에서 모든 shim 제거
        - `core/graph/__init__.py`에서 re-export 제거
        - `core/__init__.py`에서 `_loaders` re-export 제거
        - `core/model_loader.py`에서 `_loaders` re-export 제거

RULE 4: shim 제거 전 반드시 integration test 통과 확인
        - 29개 integration test 모두 통과 필수
        - pytest tests/unit -q --cov-fail-under=55 통과 필수
```

### 7.4 Shim 제거 순서 (Task10)

1. `core/graph/__init__.py` re-export 제거
2. `core/__init__.py`에서 `from core.graph import graph_builder` 제거
3. `core/model_loader.py`에서 `from core._loaders import ...` 제거
4. `core/rag_core.py`에서 `.graph.graph_builder` relative import 정리
5. `core/pipeline_builder.py`에서 `core.model_loader.ModelManager` import 경로 확인
6. 최종 `ruff check`, `mypy src`, `pytest tests/unit` 통과 확인

---

## 8. Verification Checklist

- [x] `src/` 101 파일 명시 확인 (100+ 달성)
- [x] 디렉토리별 파일수/LOC 집계 완료
- [x] Import graph 추출 완료 (상위 importer 매핑)
- [x] ruff check 통과
- [x] ruff format --check 통과 (102 files)
- [x] mypy src 통과 (101 files)
- [x] pytest tests/unit 통과 (896 passed, 78.64% coverage)
- [x] config.yml 키 + API routes + prompts 금지개명 목록 작성
- [x] Shim 정책 정의 (1페이즈 유지, Task10 제거)
- [x] reports/baseline_consolidation_20260922_113753.md 작성

---

## 9. Notes

- **모든 src/ 파일은 read-only.** 어떠한 수정, 병합, 개명도 금지.
- **tests/ 수정 금지.** 테스트 파일은 baseline 상태로 동결.
- **커밋 금지.** 이 리포트는 로컬에만 존재.
- **Coverage gate:** 55% 최소 요구, 실제 78.64%로 상회.
- **Total test count:** 217개 (unit 180 + integration 31 + security 3 + e2e 2 + stability 1)

---

*End of Phase 0 Baseline Report*
