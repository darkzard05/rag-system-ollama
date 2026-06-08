# UI Layout Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 채팅창과 PDF 미리보기창이 브라우저 화면 높이를 넘어가지 않도록 `dvh`와 `calc()`를 사용하여 레이아웃을 최적화하고 독립 스크롤을 구현합니다.

**Architecture:** 
1.  `src/ui/ui.py`에서 전역 App-Shell 스크롤을 차단하고 뷰포트 높이를 고정하는 CSS를 주입합니다.
2.  `chat.py`와 `viewer.py`에서 각 컨테이너에 전용 CSS 클래스를 부여하고, `calc(100dvh - offset)` 공식을 적용하여 높이를 동적으로 계산합니다.
3.  `constants.py`의 고정 높이 상수에 대한 의존성을 제거합니다.

**Tech Stack:** Streamlit, CSS (dvh, Flexbox, calc), Python

---

### Task 1: Global Layout Setup (Viewport Locking)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Update `inject_custom_css` to lock viewport and hide main scroll**

```python
# src/ui/ui.py (inject_custom_css 함수 내부 수정)
def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* 전체 화면 스크롤 차단 및 높이 고정 */
    .stApp {{
        height: 100dvh;
        overflow: hidden;
    }}
    /* 메인 컨테이너 패딩 최적화 */
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 0px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        height: 100%;
    }}
    /* 채팅 영역 스크롤바 스타일링 */
    .chat-scroll-container {{
        height: calc(100dvh - 12rem) !important;
        overflow-y: auto !important;
        min-height: 400px;
        padding-right: 10px;
    }}
    /* PDF 뷰어 영역 스크롤바 스타일링 */
    .viewer-scroll-container {{
        height: calc(100dvh - 10rem) !important;
        overflow-y: auto !important;
        min-height: 400px;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ui.py
git commit -m "feat(ui): lock viewport and add scroll container classes"
```

---

### Task 2: Chat Interface Optimization

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: Apply `chat-scroll-container` class to chat container**

```python
# src/ui/components/chat.py (render_chat_interface 함수 수정)
def render_chat_interface():
    # ... 이전 코드 생략 ...
    # [수정] 클래스 부여를 위해 외곽 컨테이너 추가 및 하드코딩된 height 제거 시도
    with st.container():
        st.markdown('<div class="chat-scroll-container">', unsafe_allow_html=True)
        # 기존 chat_container 내부 로직 (메시지 렌더링 등)
        # 주의: st.container(height=...) 대신 CSS 클래스 제어를 위해 일반 container 사용
        with st.container(border=False):
            # ... 메시지 루프 ...
        st.markdown('</div>', unsafe_allow_html=True)
    # ... 나머지 코드 (st.chat_input 등) ...
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/components/chat.py
git commit -m "feat(chat): apply dynamic height and independent scroll to chat"
```

---

### Task 3: PDF Viewer Optimization

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: Apply `viewer-scroll-container` class to PDF viewer container**

```python
# src/ui/components/viewer.py (render_pdf_column 함수 수정)
def render_pdf_column():
    # ... 이전 코드 (컨트롤바 등) 생략 ...
    st.markdown('<div class="viewer-scroll-container">', unsafe_allow_html=True)
    with st.container(border=False):
        _display_pdf_viewer(pdf_path, current_page, file_hash)
    st.markdown('</div>', unsafe_allow_html=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/components/viewer.py
git commit -m "feat(viewer): apply dynamic height and independent scroll to pdf viewer"
```

---

### Task 4: Clean up Constants

**Files:**
- Modify: `src/common/constants.py`

- [ ] **Step 1: Review or deprecate `CONTAINER_HEIGHT`**

```python
# src/common/constants.py
class UIConstants(IntEnum):
    """UI 관련 상수"""
    # [수정] CSS dvh 기반으로 전환되었으므로 이 값은 폴백용으로만 유지
    CONTAINER_HEIGHT = 700 
```

- [ ] **Step 2: Commit**

```bash
git add src/common/constants.py
git commit -m "refactor(ui): update height constants for css-driven layout"
```
