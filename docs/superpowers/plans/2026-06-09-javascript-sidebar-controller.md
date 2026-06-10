# JavaScript Sidebar Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a JavaScript snippet in `src/ui/ui.py` to programmatically collapse the Streamlit sidebar.

**Architecture:** Add a standalone function `inject_sidebar_closer` that uses `st.components.v1.html` to inject a JavaScript snippet. This snippet targets the sidebar and its collapse button to simulate a click if the sidebar is expanded.

**Tech Stack:** Python, Streamlit, JavaScript.

---

### Task 1: Implement `inject_sidebar_closer` in `src/ui/ui.py`

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Add `inject_sidebar_closer` function to `src/ui/ui.py`**

```python
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

- [ ] **Step 2: Verify syntax of `src/ui/ui.py`**

Run: `python -m py_compile src/ui/ui.py`
Expected: No errors.

- [ ] **Step 3: Commit the changes**

```bash
git add src/ui/ui.py
git commit -m "feat: add inject_sidebar_closer for programmatic sidebar control"
```
