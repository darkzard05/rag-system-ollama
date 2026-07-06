# Sidebar & Toolbar Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the sidebar expand button visibility and apply a modern semi-transparent blur effect to the top toolbar.

**Architecture:** Update the custom CSS injection in the main UI file to override Streamlit's default header and sidebar styles with Glassmorphism properties.

**Tech Stack:** Streamlit (Python), CSS (Glassmorphism, backdrop-filter)

---

### Task 1: Update UI Styling in `src/ui/ui.py`

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Replace the `inject_custom_css` function content**
Update the CSS to restore the toolbar with a blur effect and fix the sidebar toggle visibility.

```python
def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        """
    <style>
    /* 1. 사이드바 확장 버튼 가시성 강제 확보 (Invisible 이슈 해결) */
    [data-testid="stSidebarCollapseButton"] {
        z-index: 100000 !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border-radius: 50% !important;
    }

    /* 2. 상단 헤더 복구 및 Glassmorphism 적용 (Transparent 이슈 해결) */
    header[data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color), transparent 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 80%) !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05) !important;
        z-index: 99999 !important;
        display: flex !important; /* display: none 제거 효과 */
        visibility: visible !important;
    }

    /* 3. 불필요한 배포 버튼 등은 숨김 유지 (선택적) */
    .stAppDeployButton {
        display: none !important;
    }

    /* 4. 레이아웃 보정: 헤더와 본문 겹침 방지 */
    .block-container {
        padding-top: 4rem !important; 
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }
    
    /* 기존 스타일 유지 (Thought Process, Citations 등) */
    details.thought-expander {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--faded-text-color);
        border-radius: 8px;
        margin: 8px 0 16px 0;
        padding: 8px 12px;
    }
    .citation-highlight {
        background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
        border-bottom: 2px dashed var(--primary-color);
        padding: 0 4px;
        border-radius: 3px;
        color: var(--primary-color);
        font-weight: 600;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Commit the changes**

```bash
git add src/ui/ui.py
git commit -m "style: restore sidebar button and apply glassmorphism to header"
```

---

### Task 2: Manual Verification

- [ ] **Step 1: Check Sidebar Collapse/Expand**
1. 앱을 실행하고 사이드바를 접습니다(Collapse).
2. 화면 좌측 상단에 ">" 모양의 확장 버튼이 선명하게 보이는지 확인합니다.
3. 버튼을 클릭하여 사이드바가 정상적으로 열리는지 확인합니다.

- [ ] **Step 2: Check Toolbar Visuals**
1. 상단 툴바가 완전 투명이 아닌, 본문 내용이 부드럽게 비치는 반투명(Blur) 상태인지 확인합니다.
2. 마우스 스크롤 시 툴바 뒤로 글자가 지나가는 모습이 자연스러운지 확인합니다.
