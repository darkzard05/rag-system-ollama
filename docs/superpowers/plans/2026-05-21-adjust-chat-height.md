# Adjust Chat Interface Container Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the default `UI_CONTAINER_HEIGHT` to 500 to optimize the initial chat layout before CSS overrides.

**Architecture:** Update the global configuration variable and ensure the UI component correctly consumes it.

**Tech Stack:** Python, Streamlit

---

### Task 1: Update Global Configuration

**Files:**
- Modify: `src/common/config.py`

- [ ] **Step 1: Update `UI_CONTAINER_HEIGHT` default value**

Modify `src/common/config.py` to change the default value from 700 to 500.

```python
# src/common/config.py
# 기존: UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 700)
# 변경:
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 500)
```

- [ ] **Step 2: Verify the change via inspection**

Verify that `UI_CONTAINER_HEIGHT` is now 500.

- [ ] **Step 3: Commit the configuration change**

```bash
git add src/common/config.py
git commit -m "config: UI 컨테이너 기본 높이를 500으로 하향 조정"
```

---

### Task 2: Verify and Ensure Chat Component Usage

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Verify `UI_CONTAINER_HEIGHT` is imported and used**

Ensure `src/ui/components/chat.py` imports `UI_CONTAINER_HEIGHT` and uses it in `st.container`.

```python
# src/ui/components/chat.py
    with st.container(height=UI_CONTAINER_HEIGHT, border=False):
```

- [ ] **Step 2: Commit if any adjustment was necessary**

If no changes were needed because it was already using the variable, skip the commit or add a verification note. If a change was made:

```bash
git add src/ui/components/chat.py
git commit -m "ui: 채팅 컴포넌트가 최적화된 높이 설정을 사용하도록 확인/수정"
```

---

### Task 3: Final Verification and Cleanup

- [ ] **Step 1: Syntax check**

Run a syntax check on the modified files.

Run: `python -m py_compile src/common/config.py src/ui/components/chat.py`
Expected: No output (success)

- [ ] **Step 2: Run all tests**

Ensure no regressions were introduced.

Run: `pytest`

- [ ] **Step 3: Update local documentation (if any)**

Check if any `GEMINI.md` or other docs need updating.

- [ ] **Step 4: Commit final verification status**

```bash
git add .
git commit -m "ui: 채팅 컨테이너 높이 설정 최적화 완료"
```
