# Sidebar Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically manage sidebar state to maximize workspace efficiency: start expanded for setup, and auto-collapse once a document is loaded for analysis.

**Architecture:** Utilize `st.session_state` to track the desired sidebar state and inject a small JavaScript snippet that programmatically clicks the Streamlit sidebar collapse button when a "collapse requested" flag is detected.

**Tech Stack:** Streamlit, JavaScript (DOM manipulation), Python.

---

### Task 1: Initialize Session State and Update Page Config

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Update `st.set_page_config` to ensure `initial_sidebar_state` is "expanded"**

```python
# src/main.py around line 24
st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
    initial_sidebar_state="expanded", # Ensure this is always expanded at start
)
```

- [ ] **Step 2: Initialize a new session state variable `sidebar_auto_collapsed` to track if we've already performed the auto-collapse for the current file**

```python
# src/main.py around line 35
if "sidebar_auto_collapsed" not in st.session_state:
    st.session_state.sidebar_auto_collapsed = False
```

- [ ] **Step 3: Reset `sidebar_auto_collapsed` when a new file is uploaded**

```python
# src/main.py in on_file_upload function
def on_file_upload() -> None:
    # ... existing code ...
    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        st.session_state.sidebar_auto_collapsed = False # Reset flag for new file
        # ... rest of existing code ...
```

### Task 2: Implement JavaScript Sidebar Controller

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Add a JavaScript snippet to `inject_custom_css` (or a new function) that can collapse the sidebar by simulating a click on the collapse button**

```python
# src/ui/ui.py
def inject_sidebar_closer():
    """사이드바를 프로그램적으로 닫기 위한 JS 인젝션"""
    js = """
    <script>
        const collapseSidebar = () => {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            const collapseButton = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'true' && collapseButton) {
                collapseButton.click();
            }
        };
        // 실행 지연을 주어 Streamlit 렌더링 후 동작하도록 함
        setTimeout(collapseSidebar, 500);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)
```

### Task 3: Trigger Auto-Collapse on Document Processing Completion

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Check the RAG processing status and trigger the collapse if document is processed and auto-collapse hasn't happened yet**

```python
# src/main.py in main() function
def main() -> None:
    # ... existing setup code ...
    
    # RAG 처리가 완료되었고 아직 자동으로 닫지 않았다면 트리거
    if SessionManager.get("pdf_processed") and not st.session_state.get("sidebar_auto_collapsed", False):
        from ui.ui import inject_sidebar_closer
        inject_sidebar_closer()
        st.session_state.sidebar_auto_collapsed = True
        
    # ... rest of main logic ...
```

### Task 4: Verification and Refinement

- [ ] **Step 1: Manual verification - Start the app and ensure sidebar is open**
- [ ] **Step 2: Manual verification - Upload a PDF and wait for processing**
- [ ] **Step 3: Manual verification - Confirm sidebar closes automatically once PDF viewer appears**
- [ ] **Step 4: Manual verification - Ensure manually opening the sidebar doesn't cause it to re-close immediately**
