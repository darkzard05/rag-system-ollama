# PDF 하단 네비게이션 구현 계획서

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move PDF nav to bottom and fix the '1 / 20' layout issues.

**Architecture:** 
1. `viewer.py`: Reorder function calls and simplify column logic.
2. `ui.py`: Adjust CSS for new layout heights.

---

### Task 1: Reorder and Redesign Viewer Controls

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: Reorder `render_pdf_column`**
Move `_display_pdf_controls` after the PDF container.

- [ ] **Step 2: Simplify `_display_pdf_controls` to 3 columns**
Change columns to `[1, 2, 1]` and use "Page [X] of Y" label.

```python
# Pseudo-code change in viewer.py
c_prev, c_center, c_next = st.columns([1, 2, 1], gap="small", vertical_alignment="center")
with c_center:
    col_label, col_input, col_total = st.columns([0.6, 1, 1])
    col_label.markdown("Page")
    col_input.number_input(...)
    col_total.markdown(f"of {total_pages}")
```

---

### Task 2: CSS Layout Optimization

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Adjust `calc` for PDF viewer height**
Since nav is at bottom, the main content height might need minor adjustment to prevent double scrollbars.

---

### Task 3: Verification
- [ ] **Step 1: Manual Check**
Verify if navigation is at the bottom and looks centered with the chat input.
