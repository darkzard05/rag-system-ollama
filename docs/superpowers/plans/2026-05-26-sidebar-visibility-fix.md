# 사이드바 확장 버튼 가시성 해결 구현 계획 (Sidebar Visibility Fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit 1.54.0 버전에서 사라진 사이드바 확장 버튼을 복구하고, 스타일을 강화하며, 다른 UI 요소와의 겹침을 방지합니다.

**Architecture:** 
1. CSS 선택자를 수정하여 사이드바 내부의 '닫기' 버튼만 숨기고 '열기' 버튼은 유지합니다.
2. 사이드바 축소 시 메인 콘텐츠 영역에 60px의 여백을 주어 버튼 공간을 확보합니다.
3. JavaScript를 통해 버튼을 `body`로 이동시켜 레이아웃 클리핑을 방지합니다.

**Tech Stack:** Python, Streamlit, CSS, JavaScript

---

### Task 1: CSS 및 JavaScript 로직 수정

**Files:**
- Modify: `src/ui/ui.py`

- [ ] **Step 1: CSS 선택자 및 레이아웃 스타일 수정**

```python
# src/ui/ui.py 의 inject_custom_css 함수 내부 수정

    /* 3. 상단 안전 영역 확보 및 메인 컨테이너 최적화 */
    [data-testid="stMainBlockContainer"] {
        /* 사이드바 확장 여부에 따른 조건부 패딩 설정 */
        padding-left: """ + ("15px" if is_expanded else "60px") + """ !important;
        padding-right: 15px !important;
        padding-top: 0px !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
        overflow: hidden !important;
    }

    /* ... */

    /* 사이드바 확장 버튼 (접혔을 때 나타남): 디자인 및 위치 강화 */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        z-index: 1000001 !important;
        background-color: #007bff !important;
        border-radius: 10px !important;
        width: 44px !important;
        height: 36px !important;
        justify-content: center !important;
        align-items: center !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        cursor: pointer !important;
    }

    /* [중요] 사이드바 내부의 축소 버튼(X)만 숨김 - 확장 버튼은 유지 */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
```

- [ ] **Step 2: JavaScript 버튼 고정 로직 최신 ID 대응**

```javascript
// src/ui/ui.py 의 inject_custom_css 함수 내부 script 수정

        function fixSidebarBtn() {
            // Streamlit 원본 확장 버튼 탐색 (최신 버전 ID 우선)
            var original = document.querySelector('[data-testid="stSidebarCollapseButton"]')
                        || document.querySelector('[data-testid="stSidebarCollapsedControl"]')
                        || document.querySelector('[data-testid="collapsedControl"]');

            // 사이드바 외부(header 또는 body)에 있는 버튼만 처리하도록 필터링
            if (original && !original.closest('[data-testid="stSidebar"]')) {
                if (original.parentElement !== document.body) {
                    document.body.appendChild(original);
                }
                Object.assign(original.style, {
                    display: 'flex',
                    visibility: 'visible',
                    position: 'fixed',
                    top: '15px',
                    left: '15px',
                    zIndex: '1000001',
                    backgroundColor: '#007bff',
                    borderRadius: '10px',
                    width: '44px',
                    height: '36px',
                    justifyContent: 'center',
                    alignItems: 'center',
                    boxShadow: '0 4px 10px rgba(0,0,0,0.4)',
                    border: '2px solid rgba(255,255,255,0.2)',
                    cursor: 'pointer',
                });
            }
        }
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/ui.py
git commit -m "style: fix sidebar expand button visibility and layout overlap"
```

### Task 2: 사이드바 상태 연동 확인 및 검증

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: `main.py`에서 `is_expanded` 전달 로직 확인**

```python
# src/main.py 의 main() 함수 확인
    # [수정] PDF 업로드 상태뿐만 아니라 실제 사이드바 축소 상태를 반영하도록 플래그 체크
    # 현재는 pdf_file_path 여부로 판단하고 있으나, 
    # 사용자가 수동으로 닫았을 때도 고려하기 위해 SessionManager 상태 확인
    is_expanded = bool(SessionManager.get("pdf_file_path")) and not st.session_state.get("sidebar_collapsed", False)
    
    inject_custom_css(is_expanded=is_expanded)
```

- [ ] **Step 2: 최종 검증**
  - 앱을 실행하여 사이드바가 축소되었을 때 파란색 버튼이 (15px, 15px) 위치에 보이는지 확인.
  - PDF 컨트롤바가 버튼과 겹치지 않고 오른쪽으로 밀려나 있는지 확인.
  - 버튼 클릭 시 사이드바가 정상적으로 열리는지 확인.

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "fix: refine sidebar expansion state detection for CSS injection"
```
