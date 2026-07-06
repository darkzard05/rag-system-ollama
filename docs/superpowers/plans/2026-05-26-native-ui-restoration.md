# 구현 계획: 네이티브 UI 복구 및 접근성 표준 준수 (Native UI & Accessibility Restoration)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 그동안의 과도한 UI 커스텀(JavaScript 주입, 헤더 강제 숨김 등)을 모두 철회하고, Streamlit의 순정 기능을 기반으로 사이드바 버튼을 복구하며 접근성 경고를 해결합니다.

**Architecture:** 
1. **Clean Slate:** `ui.py`에서 불확실한 모든 JS 및 `!important` 숨김 규칙을 삭제합니다.
2. **Accessibility First:** 모든 위젯의 레이블을 생성하되, CSS로 시각적으로만 숨겨(Screen-reader only) 브라우저 경고를 해결합니다.
3. **Minimal Styling:** 이미 노출된 네이티브 버튼에 대해서만 색상 및 스타일을 보정합니다.

**Tech Stack:** Python, Streamlit, CSS

---

### Task 1: `src/ui/ui.py` 전면 재작성 (원상복구)

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: 모든 JavaScript 및 과도한 CSS 삭제**
  - 기존의 `st.markdown` 스크립트 블록과 복잡한 스타일들을 모두 제거합니다.
  - 헤더와 장식 요소들의 숨김 설정을 해제합니다.

- [ ] **Step 2: 표준 레이아웃 및 최소 스타일 적용**

```python
# src/ui/ui.py 재작성 내용
def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        """
    <style>
    /* 1. 표준 레이아웃 유지 (상단 여백은 Streamlit 기본값 사용) */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100dvh !important;
    }

    /* 2. 접근성: 레이블을 시각적으로만 숨김 (DOM에는 유지하여 ID 연결성 확보) */
    [data-testid="stWidgetLabel"] {
        clip: rect(0 0 0 0);
        clip-path: inset(50%);
        height: 1px;
        overflow: hidden;
        position: absolute;
        white-space: nowrap;
        width: 1px;
    }

    /* 3. 네이티브 버튼 스타일링 (사이드바 확장 버튼) */
    /* Streamlit 순정 버튼을 가리지 않도록 header 노출 */
    header {
        visibility: visible !important;
        background: transparent !important;
    }

    /* 확장 버튼 아이콘 및 배경 강조 */
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/ui.py
git commit -m "style: restore native UI elements and simplify CSS injection"
```

### Task 2: 컴포넌트 접근성 및 위젯 고유 키 설정

**Files:**
- Modify: `src/ui/components/sidebar.py`
- Modify: `src/ui/components/chat.py`

- [ ] **Step 1: `sidebar.py` 위젯 수정**
  - `label_visibility="collapsed"` 제거.
  - 모든 위젯(`st.file_uploader`, `st.selectbox`)에 고유한 `key` 부여 확인 및 수정.

- [ ] **Step 2: `chat.py` 위젯 수정**
  - `st.chat_input`에 명확한 레이블 부여 (시각적으로는 Task 1의 CSS가 숨김).
  - `key="main_chat_input"` 유지.

- [ ] **Step 3: Commit**

```bash
git add src/ui/components/sidebar.py src/ui/components/chat.py
git commit -m "fix: enforce accessibility standards by restoring labels and providing unique keys"
```

### Task 3: 최종 검증 및 마무리

- [ ] **Step 1: 브라우저 동작 확인**
  - 확장 버튼이 나타나는지, 클릭 시 사이드바가 정상적으로 열리는지 확인.
  - 브라우저 콘솔(F12)에서 접근성 관련 경고가 사라졌는지 확인.

- [ ] **Step 2: 체크리스트 및 설계 문서 업데이트**
- [ ] **Step 3: Commit 및 완료**
