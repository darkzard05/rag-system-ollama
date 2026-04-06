# 하단 고정(Sticky Footer) 레이아웃 정교화 계획서

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1열(PDF)과 2열(Chat)의 하단 바(컨트롤바, 입력창)가 브라우저 바닥에 완벽하게 고정(Sticky)되도록 레이아웃을 정교화합니다.

**Architecture:** 컬럼(`[data-testid="column"]`)을 Flexbox(`flex-direction: column`)로 설정하고, 상단 영역은 `flex: 1`로 확장, 하단 영역은 `position: sticky; bottom: 0;` 및 `z-index`를 부여하여 항상 최하단에 위치하게 합니다.

**Tech Stack:** Streamlit 1.54.0, Python, CSS.

---

### Task 1: 전역 Sticky CSS 강화 (src/ui/ui.py)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: Sticky Footer 전용 CSS 선택자 추가**
`src/ui/ui.py`의 `inject_custom_css` 함수를 수정하여 `[data-testid="column"] > div > div:last-child` 영역에 sticky 속성과 배경색, 보더를 부여합니다.

- [ ] **Step 2: CSS 업데이트 커밋**
`git add src/ui/ui.py; git commit -m "style: strengthen sticky footer CSS with explicit positioning and z-index"`

---

### Task 2: PDF 뷰어 하단 바 구조 최적화 (src/ui/components/viewer.py)

**Files:**
- Modify: `src/ui/components/viewer.py`

- [ ] **Step 1: 하단 컨트롤바를 별도 컨테이너로 감싸 CSS 타겟팅 보장**
컨트롤바(`st.columns`)가 CSS의 `:last-child` 선택자에 정확히 걸리도록 `st.container()`로 한 번 더 감쌉니다.

- [ ] **Step 2: PDF 뷰어 구조 변경 커밋**
`git add src/ui/components/viewer.py; git commit -m "feat: wrap PDF controls in a container for reliable sticky targeting"`

---

### Task 3: 채팅 입력창 레이아웃 일관성 확보 (src/ui/components/chat.py)

**Files:**
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: 채팅 입력창 영역 구조화**
`st.chat_input`을 `st.container()`로 감싸 하단 고정 스타일이 안정적으로 적용되게 합니다.

- [ ] **Step 2: 채팅 인터페이스 구조 변경 커밋**
`git add src/ui/components/chat.py; git commit -m "feat: wrap chat input in a container for consistent sticky styling"`

---

### Task 4: 최종 검증

- [ ] **Step 1: 레이아웃 정합성 확인**
`python tests/unit/test_ui_components.py` 실행.
