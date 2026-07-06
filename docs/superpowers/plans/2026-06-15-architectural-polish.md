# Phase 3: Architectural Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 리소스 관리 통합 및 Pydantic 기반 설정 검증 도입을 통한 아키텍처 정교화.

**Architecture:** 
1. `ModelManager`를 확장하여 `ResourcePool` 기능을 흡수하고 단일 리소스 관리 지점으로 구축.
2. `src/common/schemas.py`에 Pydantic 모델을 정의하여 `config.yml`의 무결성 보장.

**Tech Stack:** Python, Pydantic, threading, OrderedDict.

---

### Task 1: 통합 리소스 관리자 구현 및 ResourcePool 제거

**Files:**
- Modify: `src/core/model_loader.py`
- Modify: `src/core/rag_core.py`
- Delete: `src/core/resource_pool.py`
- Test: `tests/stability/test_unified_resource.py`

- [ ] **Step 1: `ModelManager`에 리소스 관리 메서드 추가 및 확장**

```python
# src/core/model_loader.py
class ModelManager:
    _resource_pool: OrderedDict[str, Any] = OrderedDict()
    _build_locks: dict[str, asyncio.Lock] = {}
    
    @classmethod
    async def get_or_build_resource(cls, key, build_fn, *args, **kwargs):
        # ResourcePool의 get_or_build 로직 이관 및 loop_integrity와 통합
        pass
```

- [ ] **Step 2: `RAGSystem` 및 관련 코드에서 `ResourcePool` 호출을 `ModelManager`로 변경**

- [ ] **Step 3: `ResourcePool` 파일 삭제 및 임포트 정리**

- [ ] **Step 4: 통합 리소스 관리 테스트 작성 및 실행**

Run: `pytest tests/stability/test_unified_resource.py`
Expected: PASS

---

### Task 2: Pydantic 기반 설정 검증 도입

**Files:**
- Create: `src/common/schemas.py`
- Modify: `src/common/config.py`
- Test: `tests/unit/test_config_validation.py`

- [ ] **Step 1: Pydantic 설정 모델 정의**

```python
# src/common/schemas.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class ModelsConfig(BaseModel):
    default_ollama: str
    max_concurrent_inference: int = Field(ge=1)
    # ...
```

- [ ] **Step 2: `src/common/config.py` 리팩토링 (검증 로직 주입)**

```python
# src/common/config.py
from common.schemas import AppConfig
_raw_config = _load_config()
config_obj = AppConfig(**_raw_config)

# 기존 전역 변수들에 할당
MAX_CONCURRENT_INFERENCE = config_obj.models.max_concurrent_inference
```

- [ ] **Step 3: 설정 검증 테스트 작성 및 실행**

Run: `pytest tests/unit/test_config_validation.py`
Expected: PASS

---

### Task 3: 최종 회귀 테스트 및 프로젝트 마무리

- [ ] **Step 1: 전체 테스트 스위트 실행**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: `graphify update .` 실행 (지식 그래프 갱신)**

- [ ] **Step 3: 최종 결과 보고 및 문서 업데이트**
