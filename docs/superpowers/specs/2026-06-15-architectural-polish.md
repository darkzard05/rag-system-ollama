# Specification: Phase 3 - Architectural Polish

**Status:** Draft
**Date:** 2026-06-15
**Phase:** 3 of 3
**Topic:** 아키텍처 정교화 (리소스 관리 통합 및 설정 검증 강화)

## 1. 개요 (Overview)
RAG 시스템의 장기 유지보수성과 안정성을 위해 분산된 리소스 관리 로직을 통합하고, 복잡한 설정값에 대한 엄격한 검증 체계를 도입합니다.

## 2. 목표 (Goals)
- `ResourcePool`과 `ModelManager`를 하나로 통합하여 메모리 관리 효율성 증대.
- Pydantic 스키마를 통한 `config.yml` 설정값 유효성 검증 (Fail-fast).
- 중복 로직 제거 및 타입 안정성(Type Safety) 강화.

## 3. 상세 설계 (Detailed Design)

### 3.1. 통합 리소스 관리자 (Unified Resource Manager)
- **대상:** `src/core/model_loader.py` (통합 대상), `src/core/resource_pool.py` (제거 대상)
- **변경 사항:**
  - `ModelManager`가 모델(LLM, Embedder)뿐만 아니라 벡터 스토어(FAISS), 리트리버(BM25)도 관리하도록 확장.
  - `get_or_build_resource` 메서드 도입: 키 기반 캐싱, LRU 정책, 중복 빌드 방지 락(Lock) 기능을 통합 제공.
  - `ResourcePool`에 존재하던 백그라운드 메모리 정리 로직(`_cleanup_memory_bg`)을 `ModelManager`로 이관.
- **이점:** 단일 지점에서 전역 시스템 리소스를 제어하므로 VRAM/RAM 고갈에 유연하게 대응 가능.

### 3.2. Pydantic 기반 설정 스키마 (Configuration Schema)
- **대상:** `src/common/config.py`, `src/common/schemas.py` (신규)
- **변경 사항:**
  - `AppConfig`, `ModelsConfig`, `RAGConfig` 등 계층적 Pydantic 모델 정의.
  - `config.yml` 로드 시 즉시 스키마 검증 수행. 잘못된 값(예: 음수 k값, 존재하지 않는 경로)이 있을 경우 명확한 에러 메시지와 함께 실행 중단.
  - 기존 전역 변수 방식(예: `MAX_CONCURRENT_INFERENCE`)을 유지하면서 내부적으로는 검증된 객체에서 값을 가져오도록 수정.
- **이점:** 사용자 오설정으로 인한 잠재적 버그 예방 및 IDE 자동완성 지원.

## 4. 데이터 흐름 (Data Flow)
1. **Config Load:** `config.yml` -> `AppConfig` (Pydantic Validation) -> Global Variables.
2. **Resource Request:** `RAGSystem` -> `ModelManager.get_or_build_resource(key, type)` -> (Cache Hit? Yes: Return | No: Build & Cache) -> Return Resource.

## 5. 검증 전략 (Verification Strategy)

### 5.1. 설정값 유효성 테스트 (`tests/unit/test_config_validation.py`)
- **방법:** 고의적으로 잘못된 `config.yml` 내용을 주입하여 Pydantic `ValidationError`가 정상적으로 발생하는지 확인.

### 5.2. 통합 리소스 관리 테스트 (`tests/stability/test_unified_resource.py`)
- **방법:** 모델과 벡터 스토어를 번갈아가며 다수 요청하여 LRU 정책에 따라 가장 오래된 리소스가 정상적으로 해제되는지 확인.

### 5.3. 전체 회귀 테스트
- **방법:** `pytest tests/` 실행을 통해 기존 기능(PDF 처리, 채팅, 성능 최적화)에 영향이 없는지 전수 조사.
