# 프로젝트 안정성 복구 및 핵심 로직 강화 설계 문서 (2026-05-19)

## 1. 개요
최근 커밋 및 작업 과정에서 유실된 핵심 로직을 복구하고, 런타임에서 발생하는 상태 접근 오류 및 스레드 안전성 문제를 근본적으로 해결하여 시스템의 신뢰성을 확보합니다.

## 2. 주요 설계 변경 사항

### 2.1 견고한 상태 접근 (Robust State Access)
- **문제:** LangGraph 노드 실행 시 `state`가 `dict` 또는 `BaseModel`로 혼용되어 `AttributeError` 발생.
- **해결:** 속성 추출 헬퍼 함수 `get_state_attr`을 도입하여 객체 타입에 관계없이 안전하게 값을 추출.
- **대상:** `src/core/graph_builder.py` 내의 모든 노드 함수 (`preprocess`, `grade_documents`, `rewrite_query`, `generate` 등).

### 2.2 전역 스레드 안전 리소스 풀 (Thread-safe ResourcePool)
- **문제:** `threading.local()`에 기반한 비동기 락은 스레드 간 공유 자원 보호에 부적합함.
- **해결:** 클래스 레벨의 전역 `threading.Lock()`을 사용하여 LRU 캐시(`_pool`) 조작을 보호.
- **이점:** 멀티 스레드 환경에서 리소스 풀의 정합성을 완벽히 보장.

### 2.3 스트리밍 핸들러 복구 및 기능 강화
- **문제:** `src/api/streaming_handler.py` 로직 유실로 인한 스트리밍 기능 마비.
- **해결:** 
    - v3.3.0 안정 버전 기반 로직 복원.
    - `StreamChunk` 스키마 내 `thought`, `status`, `performance` 필드 지원 강화.
    - TTFT(Time-to-First-Token) 최적화를 위한 첫 토큰 즉시 플러시 로직 포함.
- **인터페이스:** `TokenStreamBuffer.add_token` 등 테스트 코드와 일치하도록 메서드명 정비.

## 3. 구현 세부사항

### 3.1 GraphState 유틸리티 (src/core/graph_builder.py)
```python
def get_state_attr(state: Any, key: str, default: Any = None) -> Any:
    """dict 또는 객체 형태의 state에서 안전하게 속성 추출"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)
```

### 3.2 ResourcePool 동기화 (src/core/resource_pool.py)
```python
class ResourcePool:
    _lock = threading.Lock() # 전역 동기 락으로 교체
    
    async def get(self, file_hash: str):
        with self._lock:
            # OrderedDict 조작 보호
            ...
```

## 4. 검증 계획
1. **단위 테스트:** 실패 중인 18개 테스트를 Phase별로 해결.
2. **통합 테스트:** `scripts/test_full_pipeline.py`를 통한 전체 RAG 흐름 검증.
3. **UI 검증:** Streamlit 앱에서 실시간 스트리밍 및 타임라인 로그 표시 정상 여부 확인.
