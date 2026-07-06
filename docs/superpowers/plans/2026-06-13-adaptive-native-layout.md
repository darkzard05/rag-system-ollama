# Adaptive Native Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement independent scrolling and viewport containment using Streamlit's native `st.container(height=...)` and dynamic height detection via JS.

**Architecture:** Adaptive Native Layout - using `streamlit_js_eval` for runtime height calculation and `st.container` for scroll management.

**Tech Stack:** Streamlit 1.54.0, streamlit-js-eval, CSS.

---

### Task 1: Dynamic Height Detection and Layout Refactor

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Integrate `streamlit_js_eval` and calculate height**

Update `src/main.py` to fetch the browser height and pass it to the rendering functions.

```python
# src/main.py

def _render_app_layout(available_models: list[str] | None = None) -> None:
    from core.session import SessionManager
    from ui.components.sidebar import render_settings_content
    from streamlit_js_eval import streamlit_js_eval # Ensure this is imported

    # Get browser height
    viewport_height = streamlit_js_eval(js_expressions="window.innerHeight", key="viewport_height")
    
    # Calculate target height (default to 800 if not yet detected)
    target_height = (viewport_height - 150) if viewport_height else 800

    with st.sidebar:
        # ... (existing sidebar logic)

    col_pdf, col_chat = st.columns([1, 1], gap="medium")
    
    with col_pdf:
        # Wrap in native container with fixed height
        with st.container(height=target_height, border=False):
            from ui.components.viewer import render_pdf_column
            render_pdf_column()
            
    with col_chat:
        # Wrap in native container with fixed height
        with st.container(height=target_height, border=False):
            from ui.ui import render_left_column
            render_left_column()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat(ui): implement dynamic height detection and native containers" --no-verify
```

---

### Task 2: Simplified CSS for Global Containment

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Clean up and simplify `inject_custom_css`**

Remove the complex flex-chain CSS and focus on viewport locking and chat input scoping.

```python
# src/ui/ui.py

def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* 1. Global Viewport Lock */
    .stApp, [data-testid="stAppViewContainer"] {{
        height: 100vh !important;
        overflow: hidden !important;
    }}

    /* Remove default Streamlit padding that causes shifts */
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }}

    /* 2. Scoped Chat Input */
    [data-testid="stChatInputContainer"] {{
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 50% !important;
        width: 50% !important;
        z-index: 1000;
        background-color: var(--background-color) !important;
        border-top: 1px solid color-mix(in srgb, var(--faded-text-color) 10%, transparent);
    }}

    /* 3. Hide all other scrollbars except our containers */
    [data-testid="stMain"], [data-testid="stVerticalBlock"] {{
        overflow: hidden !important;
    }}
    
    /* Ensure the column itself doesn't scroll, only our internal container */
    [data-testid="stColumn"] {{
        overflow: hidden !important;
    }}
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): simplify CSS for global containment" --no-verify
```

---

### Task 3: Mobile and Small Screen Guard

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Update media queries for the native approach**

```css
    /* 4. Mobile Responsiveness */
    @media (max-width: 768px) {{
        .stApp, [data-testid="stAppViewContainer"] {{
            height: auto !important;
            overflow: visible !important;
        }}
        /* Revert containers to auto height on mobile */
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            height: auto !important;
        }}
        [data-testid="stChatInputContainer"] {{
            left: 0 !important;
            width: 100% !important;
        }}
    }}
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): refine mobile responsiveness for native containers" --no-verify
```

---

### Task 4: Verification

- [ ] **Step 1: Test with different browser heights**

Run: `streamlit run src/main.py`
Expected:
1. The app fills the screen regardless of window size.
2. Independent scrollbars appear ONLY inside the PDF and Chat columns.
3. Chat input is perfectly aligned to the right.
4. Resizing the window updates the height (after a brief rerun).
5. No "double scrollbar" effect on the side of the page.
