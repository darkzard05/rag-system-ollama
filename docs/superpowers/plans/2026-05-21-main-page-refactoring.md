# 메인페이지 레이아웃 및 독립 스크롤 리팩토링 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 메인페이지의 PDF 뷰어와 채팅창이 화면 높이에 고정되어 각각 독립적으로 스크롤되도록 UI를 개선합니다.

**Architecture:** CSS 변수를 활용한 뷰포트 기반 높이 계산(100dvh) 및 Streamlit App Shell의 스크롤 제어를 통해 고정된 레이아웃을 구현합니다.

**Tech Stack:** Streamlit, CSS (Flexbox, Viewport Units)

---

### Task 1: CSS 변수 도입 및 전역 레이아웃 고정 (ui.py)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: CSS 변수 정의 및 App Shell 스크롤 차단**

`inject_custom_css` 함수 내의 스타일 시트를 업데이트하여 변수 체계를 도입하고 전체 페이지 스크롤을 막습니다.

```python
def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        """
    <style>
    :root {
        --app-padding-top: 3.5rem;
        --pdf-header-height: 50px;
        --chat-footer-height: 120px;
        --content-height: calc(100dvh - var(--app-padding-top) - 1.5rem);
    }

    /* 전체 앱 스크롤 차단 */
    .stApp {
        overflow: hidden !important;
        height: 100dvh;
    }

    /* 메인 블록 여백 제거 */
    .main .block-container {
        padding-top: var(--app-padding-top) !important;
        padding-bottom: 0 !important;
        max-height: 100dvh;
    }
    ...
```

- [ ] **Step 2: 수동 검증 및 커밋**

설명: `streamlit run src/main.py` 실행 후 브라우저 창에서 전체 페이지 스크롤바가 사라졌는지 확인합니다.

```bash
git add src/ui/ui.py
git commit -m "style: introduce CSS variables and fix app shell scroll"
```

---

### Task 2: PDF 뷰어 컨테이너 높이 리팩토링 (viewer.py)

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: PDF 컨테이너 높이 수식 적용**

하드코딩된 `701` 대신 CSS 클래스나 정밀한 선택자를 사용하여 높이를 제어할 수 있도록 수정합니다.

```python
# src/ui/components/viewer.py

# 수정 전: with st.container(height=701, border=False):
# 수정 후: 고유 키를 부여하여 CSS에서 타겟팅 가능하게 함
with st.container(height=600, border=False, key="pdf_scroll_container"):
    _display_pdf_viewer(pdf_path, current_page, file_hash)
```

- [ ] **Step 2: 전역 CSS에서 PDF 스크롤 영역 타겟팅 (ui.py 업데이트)**

```python
# src/ui/ui.py

/* PDF 스크롤 컨테이너 제어 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[key="pdf_scroll_container"]) {
    height: calc(var(--content-height) - var(--pdf-header-height)) !important;
    overflow-y: auto !important;
}
```

- [ ] **Step 3: 커밋**

```bash
git add src/ui/ui.py src/ui/components/viewer.py
git commit -m "refactor: apply dynamic height to PDF viewer container"
```

---

### Task 3: 채팅 인터페이스 컨테이너 높이 리팩토링 (chat.py)

**Files:**
- Modify: `src/ui/components/chat.py`
- Modify: `src/ui/ui.py`

- [ ] **Step 1: 채팅 메시지 영역 키 부여**

```python
# src/ui/components/chat.py

# 수정 전: with st.container(height=702, border=False):
with st.container(height=600, border=False, key="chat_scroll_container"):
    ...
```

- [ ] **Step 2: 채팅 스크롤 영역 CSS 정의 (ui.py)**

```python
# src/ui/ui.py

/* 채팅 메시지 스크롤 컨테이너 제어 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[key="chat_scroll_container"]) {
    height: calc(var(--content-height) - var(--chat-footer-height)) !important;
    overflow-y: auto !important;
}
```

- [ ] **Step 3: 커밋**

```bash
git add src/ui/ui.py src/ui/components/chat.py
git commit -m "refactor: apply dynamic height to chat message container"
```

---

### Task 4: 최종 레이아웃 폴리싱 및 마진 조정

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: 사이드바 및 여백 최종 조정**

레이아웃이 꽉 찼을 때의 시각적 답답함을 해소하기 위해 여백을 미세 조정합니다.

- [ ] **Step 2: 전체 시스템 작동 확인 및 최종 커밋**

```bash
git add src/ui/ui.py
git commit -m "style: final UI polish for fixed-height layout"
```
