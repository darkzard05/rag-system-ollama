# 독립적 스크롤 레이아웃 구현 계획서

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 메인 UI의 1열(PDF 뷰어)과 2열(채팅)에 각각 독립적인 스크롤 기능을 구현하고, PDF 컨트롤바와 채팅 입력창을 각 컬럼의 하단에 고정합니다.

**Architecture:** Streamlit의 `st.container(height=...)`를 사용하여 스크롤 영역을 정의하고, 커스텀 CSS를 통해 이 컨테이너들이 뷰포트 높이(100vh)를 가득 채우도록 Flexbox 레이아웃을 강제합니다.

**Tech Stack:** Streamlit 1.54.0, Python, CSS.

---

### Task 1: 전역 CSS 업데이트 (src/ui/ui.py)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Flex 레이아웃 지원을 위한 `inject_custom_css` 수정**
`src/ui/ui.py`의 `inject_custom_css` 함수를 수정하여 다음을 수행합니다:
1. `[data-testid="stHorizontalBlock"]`에 `display: flex; height: 100vh;` 적용.
2. `[data-testid="column"]`에 `display: flex; flex-direction: column; height: 100vh; overflow: hidden;` 적용.
3. `st.container` 내부의 `stVerticalBlockBorderWrapper`를 타겟팅하여 `flex: 1; height: 0; min-height: 0;`을 부여해 가변 높이 및 스크롤을 허용.
4. 더 이상 사용되지 않는 `.pdf-container`, `.pdf-viewer-area`, `.chat-container`, `.chat-messages-area` 클래스 제거.

- [ ] **Step 2: CSS 변경사항 커밋**
`git add src/ui/ui.py && git commit -m "style: update global CSS for independent scrolling flex layout"`

---

### Task 2: PDF 뷰어 리팩토링 (하단 컨트롤바 고정)

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: PDF 뷰어를 `st.container`로 감싸고 컨트롤바를 하단으로 이동**
`src/ui/components/viewer.py`의 `_pdf_viewer_fragment`를 수정합니다:
1. 수동 `div` 태그(`pdf-container` 등)를 제거합니다.
2. `pdf_viewer` 호출을 `with st.container(height=500, border=False):` 내부로 넣습니다. (높이는 CSS에 의해 무시되고 확장됩니다.)
3. 페이지 이동 버튼 로직을 컨테이너 아래로 옮기고 `fixed-bottom-area` 클래스로 감쌉니다.

- [ ] **Step 2: PDF 뷰어 변경사항 커밋**
`git add src/ui/components/viewer.py && git commit -m "feat: move PDF control bar to bottom and enable independent scroll"`

---

### Task 3: 채팅 인터페이스 리팩토링

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: 채팅 히스토리를 `st.container`로 감싸기**
`src/ui/components/chat.py`의 `render_chat_interface`를 수정합니다:
1. 수동 `div` 태그(`chat-container`, `chat-messages-area` 등)를 제거합니다.
2. 메시지 렌더링 루프와 `streaming_placeholder`를 `with st.container(height=500, border=False):`로 감쌉니다.
3. `st.chat_input`은 컨테이너 외부에 두어 하단에 자연스럽게 고정되게 합니다.

- [ ] **Step 2: 채팅 인터페이스 변경사항 커밋**
`git add src/ui/components/chat.py && git commit -m "feat: enable independent scroll for chat history"`

---

### Task 4: 검증 및 테스트 업데이트

**Files:**
- Modify: `tests/unit/test_ui_components.py`

- [ ] **Step 1: 기존 UI 테스트 실행 및 업데이트**
기존 테스트가 버튼 생성 로직을 깨뜨리지 않았는지 확인합니다.
`python tests/unit/test_ui_components.py` 실행.

- [ ] **Step 2: 최종 커밋**
`git commit -m "chore: finalize scrolling layout implementation"`
