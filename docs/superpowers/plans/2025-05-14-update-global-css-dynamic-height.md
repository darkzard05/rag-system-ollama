# Update Global CSS for Dynamic Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the global CSS in `src/ui/ui.py` to use more aggressive Flexbox and dynamic height overrides to fix UI overflow issues.

**Architecture:** Moving from hardcoded pixel heights to viewport-based heights (`100dvh`) and ensuring Flexbox properties are applied consistently to allow proper scrolling in columns.

**Tech Stack:** Python, Streamlit, CSS

---

### Task 1: Update Global CSS in `src/ui/ui.py`

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Replace existing CSS blocks with the updated version**

In `src/ui/ui.py`, inside `inject_custom_css` function, update the CSS sections for `stMainBlockContainer`, `stColumn`, and `stVerticalBlockBorderWrapper`.

```python
    [data-testid="stMainBlockContainer"] {
        padding: 10px 20px !important;
        max-width: 100% !important;
        height: calc(100dvh - 35px) !important; /* 상단바 28px + 여백 고려 */
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
    }

    /* 4. 컬럼 내부 Flexbox 레이아웃 최적화 */
    [data-testid="stColumn"] {
        height: 100% !important;
        min-height: 0 !important;
        display: flex;
        flex-direction: column;
        overflow: hidden !important;
    }

    /* 컨테이너의 고정 높이 강제 무효화 및 부모에 맞춤 */
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
        flex-grow: 1 !important;
        height: 100% !important; 
        min-height: 0 !important;
        border: none !important;
    }
    
    /* 실제 스크롤이 발생하는 내부 블록 */
    [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        overflow-y: auto !important;
    }
```

- [ ] **Step 2: Verify the file is syntactically correct**

Run: `python -m py_compile src/ui/ui.py`
Expected: Exit code 0 (no output if successful)

- [ ] **Step 3: Commit changes**

```bash
git add src/ui/ui.py
git commit -m "style: 개선된 Flexbox 레이아웃 및 동적 높이 CSS 적용"
```
