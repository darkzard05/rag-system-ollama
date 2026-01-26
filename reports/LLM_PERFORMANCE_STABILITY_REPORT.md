# LLM 답변 생성 성능 측정 및 UI 안정화 보고서

**날짜**: 2026-01-26  
**상태**: ✅ 반영 완료 (Performance Metrics & Stability Fix)

---

## 1. 정밀 성능 측정 (Performance Metrics)

LLM 응답의 지연 시간과 처리량을 정밀하게 모니터링하기 위해 `src/core/graph_builder.py`에 로깅 로직을 추가했습니다.

### 1.1 수집 지표
*   **TTFT (Time To First Token)**: 첫 토큰 수신 시점 - 시작 시점.
*   **Thinking Duration**: 사고 과정 시작부터 종료(첫 답변 토큰 수신)까지의 시간.
*   **Answer Duration**: 첫 답변 토큰 수신부터 생성 완료까지의 시간.
*   **Throughput (TPS)**: 초당 생성된 답변 토큰 수.

### 1.2 로그 포맷 (Single-line)
실시간 모니터링 및 로그 분석기 파싱이 용이하도록 단일 행 포맷을 적용했습니다.
> `[LLM Metrics] TTFT: 0.80s | Thinking: 2.15s | Answer: 3.42s | Total: 6.37s | Tokens: 124 | Speed: 36.3 tok/s`

---

## 2. UI 안정화 및 버그 수정

### 2.1 Streamlit `update()` 충돌 해결
*   **문제**: `thought_expander.update()` 호출 시 "StreamlitAPIException: update() is not a valid command" 발생.
*   **해결**: `st.empty()` 컨테이너 전략 도입.
    *   사고 중에는 `expanded=True` 상태로 렌더링.
    *   답변 완료 시 컨테이너를 비우고 `expanded=False`인 새 expander로 교체하여 깔끔하게 정리.

### 2.2 스트리밍 무결성 프로토콜 준수
*   `astream_events`의 버전을 `v2`로 명시하여 최신 비동기 이벤트 처리 보장.
*   표준 `on_chat_model_stream` 이벤트를 의도적으로 무시하고, 백엔드에서 발행하는 커스텀 이벤트(`response_chunk`)만 수신하도록 단일화하여 **중복 출력 문제를 원천 차단**.

---

## 3. 검증 결과 요약

`tests/verify_streaming_realtime.py`를 통한 최종 검증 결과:
1.  **사고 과정 스트리밍**: 정상 (실시간 수신)
2.  **답변 본문 스트리밍**: 정상 (중복 없음)
3.  **성능 로그 출력**: 정상 (생성 완료 후 즉시 로그 기록)
4.  **UI 렌더링**: 정상 (사고 완료 후 에러 없이 답변 출력 및 박스 자동 정리)

---

## 4. 향후 관리 계획
*   **임계치 설정**: TTFT가 2.0s를 초과하거나 TPS가 5.0 이하로 떨어질 경우 경고 로그를 남기도록 확장 가능.
*   **데이터 시각화**: 수집된 메트릭 로그를 기반으로 Streamlit 대시보드에 평균 성능 그래프 추가 검토.
