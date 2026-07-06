# Update UIConstants.CONTAINER_HEIGHT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `UIConstants.CONTAINER_HEIGHT` in `src/common/constants.py` from its current value to 600.

**Architecture:** Surgical modification of a global constants file.

**Tech Stack:** Python

---

### Task 1: Update Global UI Constants

**Files:**
- Modify: `src/common/constants.py`

- [ ] **Step 1: Update `UIConstants.CONTAINER_HEIGHT`**

Modify `src/common/constants.py` to change `CONTAINER_HEIGHT` to 600.

```python
<<<<
class UIConstants(IntEnum):
    """UI 관련 상수"""

    # 채팅 및 PDF 뷰어 높이
    CONTAINER_HEIGHT = 650
    CHAT_SCROLL_HEIGHT = 650
    PDF_VIEWER_HEIGHT = 650
====
class UIConstants(IntEnum):
    """UI 관련 상수"""

    # 채팅 및 PDF 뷰어 높이
    CONTAINER_HEIGHT = 600
    CHAT_SCROLL_HEIGHT = 650
    PDF_VIEWER_HEIGHT = 650
>>>>
```

- [ ] **Step 2: Verify the change**

Run a simple check to ensure the value is updated.
Run: `python -c "from src.common.constants import UIConstants; print(UIConstants.CONTAINER_HEIGHT); assert UIConstants.CONTAINER_HEIGHT == 600"`
Expected: 600 (and no AssertionError)

- [ ] **Step 3: Commit**

```bash
git add src/common/constants.py
git commit -m "style: update UIConstants.CONTAINER_HEIGHT for better default"
```
