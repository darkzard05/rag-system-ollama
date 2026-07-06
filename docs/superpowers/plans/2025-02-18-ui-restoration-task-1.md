# Native UI & Accessibility Restoration Plan (Task 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore native Streamlit UI elements (header, sidebar button) and remove custom JS and excessive CSS from `src/ui/ui.py`.

**Architecture:** A clean rewrite of the UI entry point to use minimal, standard-compliant CSS for layout and accessibility without interfering with Streamlit's internal DOM structure via JavaScript.

**Tech Stack:** Python, Streamlit, Vanilla CSS.

---

### Task 1: `src/ui/ui.py` 전면 재작성 (원상복구)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Overwrite `src/ui/ui.py` with the restoration code**

```python
"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(원상복구 버전: 네이티브 기능 복원 및 최소 스타일 적용)
"""

from __future__ import annotations
import streamlit as st
from ui.components.chat import render_chat_interface

def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        \"\"\"
    <style>
    /* 1. 표준 레이아웃 유지 */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100dvh !important;
    }

    /* 2. 접근성: 레이블 시각적 숨김 */
    [data-testid="stWidgetLabel"] {
        clip: rect(0 0 0 0);
        clip-path: inset(50%);
        height: 1px;
        overflow: hidden;
        position: absolute;
        white-space: nowrap;
        width: 1px;
    }

    /* 3. 네이티브 헤더 및 버튼 복구 */
    header {
        visibility: visible !important;
        background: transparent !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
    }
    </style>
    \"\"\",
        unsafe_allow_html=True,
    )

def render_left_column():
    return render_chat_interface()
```

- [ ] **Step 2: Verify the file content**

Run: `cat src/ui/ui.py` (or check in editor)

- [ ] **Step 3: Update knowledge graph**

Run: `graphify update .`

- [ ] **Step 4: Commit the change**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): restore native UI elements and remove custom JS"
```
