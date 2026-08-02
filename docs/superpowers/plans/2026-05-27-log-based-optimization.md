# `app.log` 분석 기반 안정성 및 성능 최적화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `app.log`에서 발견된 벡터 캐시 보안 위반, 중복 임베딩 수행, 테스트 로그 오염 문제를 해결하여 시스템의 안정성과 인덱싱 성능을 개선합니다.

**Architecture:** 
1. `RAGOrchestrator` 파이프라인에서 `SemanticChunker`가 반환하는 임베딩 벡터를 `VectorStore` 생성 시 직접 전달하여 중복 계산을 제거합니다.
2. FAISS 인덱스(디렉토리)에 대한 무결성 검증 방식을 개별 파일 검증으로 세분화하여 `CacheIntegrityError` 오탐을 방지합니다.
3. 로깅 설정을 보완하여 테스트 환경의 Mock 에러가 운영 로그에 유입되는 것을 차단합니다.

**Tech Stack:** Python, FAISS, LangChain, Logging

---

### Task 1: 중복 임베딩 제거 (성능 최적화)

**Files:**
- Modify: `src/core/rag_core.py`
- Test: `tests/unit/test_rag_performance.py` (신규 생성)

- [ ] **Step 1: `RAGOrchestrator.build_pipeline` 수정**
  - `split_documents`에서 반환되는 두 번째 인자(벡터 리스트)를 변수에 할당하고, 이를 `create_vector_store`에 인자로 전달합니다.

```python
# src/core/rag_core.py 수정 예시
# ...
        # 3. 청크 분할 (Async)
        doc_splits, vectors = await split_documents(
            documents, embedder=embedder, session_id=self.session_id
        )
# ...
        # 4. 벡터 스토어 생성 (Sync)
        vector_store = create_vector_store(doc_splits, embedder, vectors=vectors)
# ...
```

- [ ] **Step 2: 단위 테스트 작성**
  - `create_vector_store` 호출 시 `embedder.embed_documents`가 호출되지 않는지(벡터가 전달되었을 때) 검증하는 테스트를 작성합니다.

- [ ] **Step 3: 테스트 실행 및 확인**
  - `pytest tests/unit/test_rag_performance.py` 실행하여 "전달된 벡터가 없어 임베딩을 다시 수행합니다." 경고가 발생하지 않는지 확인합니다.

---

### Task 2: 벡터 캐시 무결성 검증 로직 보완 (안정성)

**Files:**
- Modify: `src/cache/vector_cache.py`
- Modify: `src/security/cache_security.py`

- [ ] **Step 1: 디렉토리 해시 계산 지원 추가**
  - `CacheSecurityManager.compute_file_hash`가 디렉토리 경로를 받았을 때 내부 파일들의 해시를 결합하여 계산하거나, FAISS의 주요 파일(`index.faiss`)만 검증하도록 수정합니다.

- [ ] **Step 2: `VectorStoreCache` 무결성 검증 대상 구체화**
  - FAISS 인덱스 디렉토리 자체가 아닌, 그 내부의 실제 데이터 파일들을 검증 대상으로 명시합니다.

- [ ] **Step 3: 캐시 재생성 시나리오 테스트**
  - 인위적으로 해시를 변경한 후 `CacheIntegrityError`가 적절히 발생하는지, 그리고 정상적인 경우에는 발생하지 않는지 확인합니다.

---

### Task 3: 로깅 오염 방지 및 환경 분리 (가독성)

**Files:**
- Modify: `src/main.py`
- Modify: `src/common/logging_config.py`

- [ ] **Step 1: 메인 엔트리포인트 보호**
  - `src/main.py`에서 테스트 환경(`pytest` 실행 중 등)인지 확인하여 Mock 객체가 운영 로깅 설정을 건드리지 않도록 가드 코드를 추가합니다.

- [ ] **Step 2: 로그 레벨 및 포맷 미세 조정**
  - `CRITICAL` 보안 로그 발생 시 원인 정보(예: 어떤 파일의 해시가 다른지)를 더 상세히 남기도록 개선합니다.

- [ ] **Step 3: 최종 통합 확인**
  - 앱을 실행하고 인덱싱을 수행한 후 `logs/app.log`를 확인하여 중복 임베딩 경고와 불필요한 캐시 삭제 로그가 사라졌는지 검증합니다.
