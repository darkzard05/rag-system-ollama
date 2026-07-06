# Refined Native Layout Implementation Plan (Scroll Fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore independent scrolling by allowing Streamlit's native container to manage its own overflow, while keeping the main viewport locked.

**Architecture:** Minimal CSS + Native Scroll - removal of aggressive `overflow: hidden` on vertical blocks and refining the height calculation offset.

**Tech Stack:** Streamlit 1.54.0, streamlit-js-eval, CSS.

---

### Task 1: Fix CSS Over-targeting

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Remove aggressive overflow locks**

We must stop setting `overflow: hidden` on `stVerticalBlock` and `stColumn` to allow the native containers to function.

```python
# src/ui/ui.py

def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* 1. Only lock the outermost app container */
    .stApp, [data-testid="stAppViewContainer"] {{
        height: 100vh !important;
        overflow: hidden !important;
    }}

    /* 2. Block container padding refinement */
    .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }}

    /* 3. Scoped Chat Input (Keep this as it's necessary) */
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

    /* 4. Ensure the main block is hidden to prevent double scrolls */
    [data-testid="stMain"] {{
        overflow: hidden !important;
    }}
    
    /* REMOVED: [data-testid="stVerticalBlock"] and [data-testid="stColumn"] overflow locks */
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "fix(ui): remove aggressive CSS overflow locks to restore native scrolling" --no-verify
```

---

### Task 2: Refine Native Height Calculation

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Increase height offset and add safety margin**

Increase the `HEADER_OFFSET` to ensure the containers fit within the viewport without clipping.

```python
# src/main.py

def _render_app_layout(available_models: list[str] | None = None) -> None:
    # ...
    # Get browser height dynamically
    viewport_height = streamlit_js_eval(js_expressions="window.innerHeight", key="viewport_height")
    
    # Increase offset from 150 to 180 to account for padding and chat input safety
    # Ensure target_height is never too large
    if viewport_height:
        target_height = max(400, viewport_height - 180)
    else:
        target_height = 700 # Safer default for initial load
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "fix(ui): refine dynamic height calculation with safer offsets" --no-verify
```

---

### Task 3: Verification

- [ ] **Step 1: Run and verify scrollbars**

Run: `streamlit run src/main.py`
Expected:
1. No scrollbar on the far right (browser level).
2. Independent scrollbars appear inside the PDF column and the Chat column when content exceeds height.
3. Content is fully visible and not clipped at the bottom.
4. Chat input is correctly positioned.
