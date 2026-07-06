# UI Layout Overflow Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the UI layout where "Preview" and "Chat" columns overflow the screen by replacing hardcoded pixel heights with dynamic CSS-based sizing.

**Architecture:** 
1.  Inject CSS via `ui.py` to override Streamlit's inline `height` styles on containers, forcing them to fill available space using Flexbox.
2.  Use `calc(100dvh - [offset])` to ensure the main container exactly fits the viewport.
3.  Set internal containers to `flex-grow: 1` and `height: 100%` with `overflow-y: auto`.

**Tech Stack:** Streamlit, CSS (Flexbox, dvh), Python

---

### Task 1: Update Global CSS for Dynamic Height

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Modify CSS in `inject_custom_css`**

Replace hardcoded styles with more aggressive Flexbox and height overrides.

```python
    # src/ui/ui.py (inject_custom_css 함수 내부)
    """
    [기존 코드 일부 수정]
    """
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

- [ ] **Step 2: Commit changes**

```bash
git add src/ui/ui.py
git commit -m "style: 개선된 Flexbox 레이아웃 및 동적 높이 CSS 적용"
```

---

### Task 2: Adjust PDF Viewer Container Height

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: Reduce hardcoded height to a symbolic value**

Since Task 1's CSS will override the height to 100%, we keep `height` in `st.container` to trigger Streamlit's scrollable container logic, but the actual pixels will be managed by CSS.

```python
    # src/ui/components/viewer.py
    # 기존: with st.container(height=800, border=False):
    # 변경:
    with st.container(height=500, border=False): # CSS가 100%로 덮어씌울 것이므로 상징적인 값 사용
        _display_pdf_viewer(pdf_path, current_page, file_hash)
```

- [ ] **Step 2: Commit changes**

```bash
git add src/ui/components/viewer.py
git commit -m "ui: PDF 뷰어 컨테이너 높이 설정 최적화"
```

---

### Task 3: Adjust Chat Interface Container Height

**Files:**
- Modify: `src/ui/components/chat.py`
- Modify: `src/common/config.py`

- [ ] **Step 1: Update `config.py` default height**

Reduce the default `UI_CONTAINER_HEIGHT` to ensure it doesn't push the layout if CSS fails to load for a split second.

```python
# src/common/config.py
# 기존: UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 700)
# 변경:
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 500)
```

- [ ] **Step 2: Verify `chat.py` usage**

Ensure `render_chat_interface` uses this variable.

```python
# src/ui/components/chat.py
    with st.container(height=UI_CONTAINER_HEIGHT, border=False):
```

- [ ] **Step 3: Commit changes**

```bash
git add src/common/config.py src/ui/components/chat.py
git commit -m "ui: 채팅 컨테이너 높이 설정 최적화"
```

---

### Task 4: Final Verification

- [ ] **Step 1: Manual verification**
- 브라우저를 띄워 PDF 미리보기와 채팅창이 화면에 꽉 차는지, 잘리는 부분이 없는지 확인.
- 창 크기를 줄였을 때 입력창이 가려지지 않고 유지되는지 확인.
