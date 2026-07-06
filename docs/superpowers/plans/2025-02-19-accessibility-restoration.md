# Native UI & Accessibility Restoration - Task 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve UI accessibility by removing `label_visibility="collapsed"` and ensuring all widgets have unique `key` parameters.

**Architecture:** Modify Streamlit component files to expose labels (to be hidden via CSS in another task) and provide unique identifiers for session state management.

**Tech Stack:** Python, Streamlit

---

### Task 2.1: Update `src/ui/components/sidebar.py`

**Files:**
- Modify: `src/ui/components/sidebar.py`

- [ ] **Step 1: Remove `label_visibility="collapsed"` and check keys**

I will remove `label_visibility="collapsed"` from all widgets in `_render_settings_internal`. I will also verify that `vram_btn` and `reset_btn` have unique keys (which they do).

```python
# Before (example)
st.file_uploader(
    "PDF 파일 업로드",
    type="pdf",
    key="pdf_uploader",
    on_change=file_uploader_callback,
    disabled=is_generating,
    label_visibility="collapsed" # If it were here
)

# After
st.file_uploader(
    "PDF 파일 업로드",
    type="pdf",
    key="pdf_uploader",
    on_change=file_uploader_callback,
    disabled=is_generating,
)
```

*(Note: Looking at the read file, `label_visibility="collapsed"` is NOT actually present in the provided code snippet, but the instructions say to remove it if present. I will double check the whole file.)*

- [ ] **Step 2: Commit changes**

```bash
git add src/ui/components/sidebar.py
git commit -m "chore: remove label_visibility and verify keys in sidebar"
```

### Task 2.2: Update `src/ui/components/chat.py`

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Ensure `st.chat_input` has `key="main_chat_input"`**

I will verify `st.chat_input` uses the correct key.

```python
# In render_chat_interface
user_query = st.chat_input(input_placeholder, disabled=is_generating, key="main_chat_input")
```

- [ ] **Step 2: Commit changes**

```bash
git add src/ui/components/chat.py
git commit -m "chore: ensure unique key for chat input"
```

### Task 2.3: Verification

- [ ] **Step 1: Run linting/type checking**

Run: `ruff check src/ui/components/sidebar.py src/ui/components/chat.py`
Expected: No errors related to the changes.
