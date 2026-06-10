# 디자인 명세서: 성능 지표 UI 개선 (Data-centric Table)

**날짜:** 2026-06-10
**상태:** 승인됨 (구현 준비 중)
**목표:** 좁은 팝업 공간에서 정보가 잘리는 문제를 해결하고, 현대적인 대시보드 스타일의 정돈된 성능 지표를 제공함.

## 1. 개요
현재 `st.metric` 컴포넌트는 폰트 크기가 크고 여백이 많아 정보 밀도가 낮음. 이를 커스텀 HTML/CSS 테이블 구조로 교체하여 가독성을 높이고 시각적 상태 피드백을 추가함.

## 2. 디자인 요구사항
- **가독성:** 폰트 크기를 기존 16~18px에서 13px 수준으로 축소하여 모든 정보가 한눈에 들어오게 함.
- **밀도:** 불필요한 상하 여백을 줄여 팝업 크기를 최적화함.
- **상태 피드백:** 수치에 따라 (Excellent, Stable, Poor)와 같은 직관적인 레이블 및 아이콘 제공.
- **테마 호환성:** Streamlit의 다크/라이트 모드 테마 변수(`var(--text-color)`, `var(--primary-color)`)를 적극 활용.

## 3. 기술적 세부 사양

### 3.1 CSS 명세 (src/ui/ui.py 추가분)
```css
.perf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
    font-family: inherit;
}
.perf-row {
    border-bottom: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 90%);
}
.perf-row:last-child {
    border-bottom: none;
}
.perf-label {
    color: var(--faded-text-color);
    padding: 8px 0;
    width: 40%;
    text-align: left;
}
.perf-value {
    font-weight: 600;
    text-align: right;
    padding: 8px 12px;
    color: var(--text-color);
}
.perf-status {
    width: 35%;
    text-align: right;
    font-size: 11px;
    font-weight: 500;
}
.status-excellent { color: #2ecc71; }
.status-stable { color: var(--primary-color); }
.status-poor { color: #e74c3c; }
```

### 3.2 성능 상태 판별 기준 (Thresholds)
| 지표 | Excellent (✨) | Stable (🟢) | Poor (⚠️) |
| :--- | :--- | :--- | :--- |
| **소요 시간 (Total)** | < 1.5s | < 3.5s | >= 3.5s |
| **생성 속도 (TPS)** | > 40 tok/s | > 20 tok/s | <= 20 tok/s |

### 3.3 HTML 구조 (src/ui/components/chat.py 구현 대상)
```html
<table class="perf-table">
  <tr class="perf-row">
    <td class="perf-label">⏱️ 응답 지연</td>
    <td class="perf-value">1.2s</td>
    <td class="perf-status status-excellent">✨ Excellent</td>
  </tr>
  <tr class="perf-row">
    <td class="perf-label">⚡ 생성 속도</td>
    <td class="perf-value">45.2 t/s</td>
    <td class="perf-status status-stable">🟢 Stable</td>
  </tr>
  <tr class="perf-row">
    <td class="perf-label">🎟️ 토큰 사용</td>
    <td class="perf-value">124 / 542</td>
    <td class="perf-status">666 total</td>
  </tr>
</table>
```

## 4. 변경 대상 파일
1. `src/ui/ui.py`: `inject_custom_css` 함수 내에 스타일 추가.
2. `src/ui/components/chat.py`: `render_message` 함수 내 성능 지표 출력 로직 수정.

## 5. 자가 검토 (Self-Review)
- [x] Placeholder 제거됨
- [x] 내부 일관성 확인됨
- [x] 모바일/좁은 화면 대응 여부 확인됨
- [x] Streamlit 테마 변수 사용 여부 확인됨
