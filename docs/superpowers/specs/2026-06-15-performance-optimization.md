# Specification: Phase 2 - Pipeline Performance Optimization

**Status:** Draft
**Date:** 2026-06-15
**Phase:** 2 of 3
**Topic:** RAG 파이프라인 성능 최적화 (추론 병목 해제 및 인용 처리 고속화)

## 1. 개요 (Overview)
현재 RAG 시스템의 최대 병목 지점인 직렬화된 추론 구조와 비효율적인 인용 처리 알고리즘을 개선하여, 전체 응답 속도를 40% 이상 향상시키고 멀티유저 대응 능력을 확보합니다.

## 2. 목표 (Goals)
- 동시 추론 수를 1개에서 4개로 상향하여 처리량(Throughput) 증대.
- 인용 처리 복잡도를 $O(N \times M)$에서 $O(N)$으로 개선하여 생성 후 지연 시간 제거.
- 최적화 전후의 정량적 성능 지표(TTFT, Total Latency) 비교 보고.

## 3. 상세 설계 (Detailed Design)

### 3.1. 추론 병목 해제 (Parallel Inference)
- **파일:** `src/common/config.py`
- **변경 사항:** 
  - `MAX_CONCURRENT_INFERENCE` 기본값을 `1`에서 `4`로 수정.
  - VRAM 사용량이 낮은 모델(예: 4B 이하) 환경에서 최적의 성능을 발휘하도록 설정.
- **기대 효과:** 배치 평가 및 다중 사용자 요청 처리 속도 향상.

### 3.2. 고속 인용 처리 (O(N) Citation Matching)
- **파일:** `src/common/utils.py`
- **변경 사항:**
  - `apply_tooltips_to_response` 함수 내부의 이중 루프 제거.
  - 문서 리스트(`documents`)를 `(page, section)`을 키로 하는 **해시 맵(Dictionary)**으로 사전 변환(Pre-processing).
  - 답변 내 인용 정보를 찾을 때마다 해시 맵에서 즉시 검색($O(1)$)하여 전체 복잡도를 $O(N)$으로 단축.
- **기대 효과:** 답변 생성 직후 발생하는 8~12초의 인용 매칭 지연을 0.5초 이내로 단축.

## 4. 검증 및 벤치마크 전략

### 4.1. 정량적 지표 측정
- **측정 항목:** 
  - Time to First Token (TTFT)
  - Total Response Latency (생성 + 인용 처리 포함)
  - CPU/GPU Peak Usage
- **방법:** `scripts/bench_ui_render_v2.py` 또는 유사한 벤치마크 스크립트를 사용하여 최적화 전/후 데이터 비교.

## 5. 예외 처리 (Error Handling)
- 동시 추론 증가로 인한 VRAM 부족 시, `ModelManager`의 LRU 캐시가 작동하여 미사용 모델을 해제하도록 보장.
- 인용 매칭 실패 시(해시 맵에 키 없음), 기존처럼 안전한 폴백 메시지 출력.
