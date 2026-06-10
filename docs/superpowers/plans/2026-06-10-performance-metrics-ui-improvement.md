# 성능 지표 UI 개선 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 성능 지표 팝업 내의 크고 불필요한 `st.metric` 컴포넌트를 현대적인 HTML/CSS 테이블 구조로 교체하여 가독성과 정보 밀도를 개선함.

**Architecture:** `src/ui/ui.py`에 전역 CSS 스타일을 주입하고, `src/ui/components/chat.py`에서 성능 데이터를 분석하여 적절한 상태(status)와 함께 HTML 테이블을 렌더링함.

**Tech Stack:** Streamlit, CSS, Python (HTML string rendering)

---

### Task 1: 커스텀 CSS 추가

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: `inject_custom_css` 함수에 성능 지표용 CSS 추가**

```python
# src/ui/ui.py 내의 inject_custom_css 함수 하단 스타일 추가
    /* 8. 성능 지표(Performance Metrics) 커스텀 테이블 스타일 */
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 5px;
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
    .status-excellent { color: #2ecc71 !important; }
    .status-stable { color: var(--primary-color) !important; }
    .status-poor { color: #e74c3c !important; }
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "style: Add custom CSS for performance metrics table"
```

---

### Task 2: 성능 상태 판별 로직 구현

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: 성능 상태(Excellent, Stable, Poor)를 판별하는 헬퍼 함수 추가**

```python
def _get_performance_status(total_time: float, tps: float) -> dict[str, dict[str, str]]:
    """수치에 따른 성능 상태와 클래스명을 반환합니다."""
    # Latency (Total Time)
    if total_time < 1.5:
        latency = {"label": "✨ Excellent", "class": "status-excellent"}
    elif total_time < 3.5:
        latency = {"label": "🟢 Stable", "class": "status-stable"}
    else:
        latency = {"label": "⚠️ Poor", "class": "status-poor"}

    # TPS (Tokens per second)
    if tps > 40:
        throughput = {"label": "🚀 Fast", "class": "status-excellent"}
    elif tps > 20:
        throughput = {"label": "🟢 Normal", "class": "status-stable"}
    else:
        throughput = {"label": "🐢 Slow", "class": "status-poor"}

    return {"latency": latency, "throughput": throughput}
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/components/chat.py
git commit -m "feat: Add performance status evaluation logic"
```

---

### Task 3: 성능 지표 렌더링 로직 교체

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: `render_message` 내의 `st.metric` 코드를 HTML 테이블로 교체**

```python
# src/ui/components/chat.py: render_message 함수 내부 수정

                if metrics:
                    total = metrics.get("total_time", 0)
                    tps = metrics.get("tps", metrics.get("tokens_per_second", 0))
                    in_tok = metrics.get("input_token_count", 0)
                    out_tok = metrics.get(
                        "token_count", metrics.get("output_token_count", 0)
                    )
                    
                    # 성능 상태 판별
                    status = _get_performance_status(total, tps)
                    
                    st.markdown("**📊 성능 지표**")
                    
                    perf_html = f"""
                    <table class="perf-table">
                        <tr class="perf-row">
                            <td class="perf-label">⏱️ 응답 지연</td>
                            <td class="perf-value">{total:.1f}s</td>
                            <td class="perf-status {status['latency']['class']}">{status['latency']['label']}</td>
                        </tr>
                        <tr class="perf-row">
                            <td class="perf-label">⚡ 생성 속도</td>
                            <td class="perf-value">{tps:.1f} t/s</td>
                            <td class="perf-status {status['throughput']['class']}">{status['throughput']['label']}</td>
                        </tr>
                        <tr class="perf-row">
                            <td class="perf-label">🎟️ 토큰 사용</td>
                            <td class="perf-value">{in_tok} / {out_tok}</td>
                            <td class="perf-status">{in_tok + out_tok} total</td>
                        </tr>
                    </table>
                    """
                    st.markdown(perf_html, unsafe_allow_html=True)
```

- [ ] **Step 2: 최종 UI 확인 (Manual Verification)**
- Streamlit 앱을 실행하여 팝업 내 테이블이 의도한 대로 렌더링되는지 확인.

- [ ] **Step 3: Commit**

```bash
git add src/ui/components/chat.py
git commit -m "ui: Replace st.metric with custom HTML performance table"
```
