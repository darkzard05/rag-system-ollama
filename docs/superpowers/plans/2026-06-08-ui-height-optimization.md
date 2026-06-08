# UI Height Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the UI overflow issue by adjusting CSS height offsets and minimum height constraints in the chat and viewer windows.

**Architecture:** Approach 1 - Offset Optimization. Increase vertical offsets in CSS `calc()` functions and lower `min-height` to ensure visibility within the viewport.

**Tech Stack:** Streamlit, CSS, Python

---

### Task 1: Update Global UI Constants

**Files:**
- Modify: `src/common/constants.py`

- [ ] **Step 1: Update `UIConstants.CONTAINER_HEIGHT`**

Modify `src/common/constants.py` to change `CONTAINER_HEIGHT` from 700 to 600. This provides a safer default for the Python-side height parameter, although it is often overridden by CSS.

```python
class UIConstants(IntEnum):
    """UI 관련 상수 기초"""

    # 채팅 및 PDF 뷰어 기본 높이 (실제 화면 렌더링은 ui.py의 CSS calc(vh)가 덮어씌워 반응형으로 동작함)
    CONTAINER_HEIGHT = 600
```

- [ ] **Step 2: Commit**

```bash
git add src/common/constants.py
git commit -m "style: update UIConstants.CONTAINER_HEIGHT for better default"
```

---

### Task 2: Optimize CSS Offsets and Constraints

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Adjust Chat Scroll Container CSS**

Update `.chat-scroll-container` to increase height offset from `12rem` to `15rem` and decrease `min-height` from `400px` to `200px`.

```css
    /* 채팅 영역 스크롤바 스타일링 */
    .chat-scroll-container {
        height: calc(100dvh - 15rem) !important;
        overflow-y: auto !important;
        min-height: 200px;
        padding-right: 10px;
    }
```

- [ ] **Step 2: Adjust Viewer Scroll Container CSS**

Update `.viewer-scroll-container` to increase height offset from `10rem` to `13rem` and decrease `min-height` from `400px` to `200px`.

```css
    /* PDF 뷰어 영역 스크롤바 스타일링 */
    .viewer-scroll-container {
        height: calc(100dvh - 13rem) !important;
        overflow-y: auto !important;
        min-height: 200px;
    }
```

- [ ] **Step 3: Adjust Thought Container Max-Height**

Update `.thought-container` to decrease `max-height` from `300px` to `250px`.

```css
    .thought-container {
        border-left: 3px solid var(--primary-color);
        padding: 10px 15px;
        margin-top: 12px;
        font-size: 0.85em;
        color: var(--text-color);
        opacity: 0.85;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        white-space: pre-wrap;
        max-height: 250px;
        overflow-y: auto;
        background-color: color-mix(in srgb, var(--background-color) 50%, transparent);
        border-radius: 0 4px 4px 0;
    }
```

- [ ] **Step 4: Commit**

```bash
git add src/ui/ui.py
git commit -m "style: optimize scroll container heights and min-heights"
```

---

### Task 3: Verification

- [ ] **Step 1: Manual Visual Check**

Run the Streamlit application and verify that:
1. The chat input is fully visible without scrolling the main page.
2. The PDF viewer navigation buttons are visible.
3. On a small browser window (height ~700px), the containers shrink appropriately without overflowing the main viewport.

Run: `streamlit run src/main.py` (assuming standard entry point)
Verification: Visually confirm no bottom overflow.
