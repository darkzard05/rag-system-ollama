# Independent Column Scrolling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve independent scrolling for PDF preview and Chat columns within a fixed 100vh viewport, with a scoped chat input.

**Architecture:** Flex-Container Strategy (Modern) - using `flex-grow: 1` and `min-height: 0` on intermediate Streamlit containers to force child columns to handle overflow.

**Tech Stack:** Streamlit (Python), CSS Flexbox.

---

### Task 1: Viewport and Flex Chain Setup

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Update `inject_custom_css` for Viewport Locking**

Replace the existing CSS in `src/ui/ui.py` with the following flex-chain structure.

```python
def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* 1. Viewport Locking */
    .stApp {{
        height: 100vh !important;
        overflow: hidden !important;
    }}

    .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
    }}

    /* 2. Flex Chain - Intermediate containers */
    /* Streamlit structure: .block-container > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] */
    .block-container > div {{
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        min-height: 0;
    }}

    [data-testid="stVerticalBlock"] {{
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        min-height: 0;
    }}

    [data-testid="stHorizontalBlock"] {{
        display: flex;
        flex-grow: 1;
        min-height: 0;
    }}
```

- [ ] **Step 2: Define Independent Column Scrolling**

Add the column-specific CSS to the same `inject_custom_css` function.

```css
    /* 3. Independent Columns */
    [data-testid="stColumn"] {{
        height: 100% !important; 
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 5px;
        scroll-behavior: smooth;
        display: flex;
        flex-direction: column;
    }}
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): implement flex-chain for independent column scrolling" --no-verify
```

---

### Task 2: Scoped Chat Input Styling

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Update `stChatInputContainer` CSS**

Scope the chat input to the right 50% of the screen.

```css
    /* 4. Scoped Chat Input */
    [data-testid="stChatInputContainer"] {{
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 50% !important; /* Start from middle */
        width: 50% !important; /* Take right half */
        padding-bottom: 1.5rem !important;
        padding-top: 0.5rem !important;
        background-color: var(--background-color) !important;
        z-index: 100;
        border-top: 1px solid color-mix(in srgb, var(--faded-text-color) 10%, transparent);
    }}

    /* Ensure column has enough bottom space so messages aren't hidden by the input */
    [data-testid="stColumn"]:last-child > div {{
        padding-bottom: 120px !important;
    }}
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): scope chat input to the right column" --no-verify
```

---

### Task 3: Mobile Responsiveness Guard

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Add Media Query for Mobile Layout**

On small screens where Streamlit stacks columns, we should revert to standard scrolling.

```css
    /* 5. Mobile Responsiveness */
    @media (max-width: 768px) {{
        .stApp, .block-container {{
            height: auto !important;
            overflow: visible !important;
        }}
        [data-testid="stColumn"] {{
            height: auto !important;
            overflow: visible !important;
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
git commit -m "feat(ui): add mobile responsiveness for independent scrolling" --no-verify
```

---

### Task 4: Verification

- [ ] **Step 1: Run the application and verify visually**

Run: `streamlit run src/main.py`
Expected: 
1. The page does not have a global scrollbar.
2. PDF viewer scrolls independently.
3. Chat history scrolls independently.
4. Chat input stays at the bottom of the chat column.
5. Resizing the window to a small width (mobile) reverts to a stacked layout with normal scrolling.
