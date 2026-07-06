# Fix Security and Quality Issues in Chat Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix XSS vulnerabilities and broad exception handling in `src/ui/components/chat.py`.

**Architecture:** Use `html.escape` to sanitize LLM-generated content before injecting it into HTML templates rendered by Streamlit. Refine exception handling to be more specific.

**Tech Stack:** Python, Streamlit, `html` (stdlib).

---

### Task 1: Research and Reproduction

**Files:**
- Create: `tests/security/test_xss_protection.py`
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Write the reproduction test for XSS**

```python
import unittest
from html import escape

def test_xss_vulnerability():
    payload = "</div><script>alert('xss')</script>"
    # This is what happens in chat.py
    thought_html = f'<div class="thought-container">{payload}</div>'
    
    # If not escaped, the script tag is present and the div is closed prematurely
    assert "</div><script>" in thought_html
    
    # The fix should look like this:
    escaped_payload = escape(payload)
    fixed_html = f'<div class="thought-container">{escaped_payload}</div>'
    assert "&lt;/div&gt;&lt;script&gt;" in fixed_html
    assert "<script>" not in fixed_html
```

- [ ] **Step 2: Identify the broad exception in chat.py**

Line 249 in `src/ui/components/chat.py`:
```python
                            except Exception:
                                pass
```

---

### Task 2: Implement Security Fixes

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Add `import html` to `src/ui/components/chat.py`**

- [ ] **Step 2: Sanitize `thought` in `render_message`**

```python
        if thought and thought.strip():
            escaped_thought = html.escape(thought)
            st.markdown(
                f"""
                <details class="thought-expander">
                    <summary>{MSG_THINKING[:-3]} 완료</summary>
                    <div class="thought-container">{escaped_thought}</div>
                </details>
                """,
                unsafe_allow_html=True,
            )
```

- [ ] **Step 3: Sanitize `thought_acc` and `current_status` in `render_chat_interface`**

Ensure `thought_acc` is escaped before being put into `thought_html`.
Ensure `chunk.status` is escaped before being put into `current_status`.

---

### Task 3: Implement Quality Fixes

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Replace broad exception handling**

```python
                            except (ValueError, TypeError, IndexError, AttributeError):
                                pass
```
Or use `contextlib.suppress(ValueError, TypeError, IndexError, AttributeError)`.

---

### Task 4: Verification

- [ ] **Step 1: Run the new security test**
- [ ] **Step 2: Manually verify (mental check) that all `unsafe_allow_html=True` calls have sanitized inputs**
- [ ] **Step 3: Run all existing unit tests to ensure no regressions**

Run: `pytest tests/unit/test_ui_components.py`
Run: `pytest tests/unit/test_core_utils.py`
