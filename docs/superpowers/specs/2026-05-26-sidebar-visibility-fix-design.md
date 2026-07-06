# 설계 문서: 사이드바 확장 버튼 가시성 및 레이아웃 개선 (Sidebar Visibility Fix)

이 문서는 사이드바 축소 상태에서 확장 버튼이 보이지 않는 문제를 해결하고, 다른 UI 요소와의 겹침을 방지하기 위한 설계안을 담고 있습니다.

## 1. 문제 분석 (Problem Analysis)
- **원인:** Streamlit 1.54.0 버전에서 사이드바의 '열기' 버튼과 '닫기' 버튼이 동일한 `data-testid="stSidebarCollapseButton"`을 공유함.
- **현상:** `src/ui/ui.py`에서 해당 ID에 `display: none !important`를 적용하여 확장 버튼이 완전히 제거됨.
- **겹침 문제:** 버튼이 노출되더라도 헤더 높이가 0으로 설정되어 있어, 좌상단에 위치한 PDF 컨트롤바와 물리적으로 겹쳐 사용이 어려움.

## 2. 해결 방안 (Proposed Solution)

### 2.1 CSS 선택자 범위 제한
- 사이드바 내부에 있는 '닫기' 버튼만 선택적으로 숨깁니다.
- `[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display: none !important; }`

### 2.2 확장 버튼 스타일 및 위치 조정
- 사이드바 외부에 노출되는 확장 버튼에 명확한 스타일을 부여합니다.
- **위치:** `top: 15px; left: 15px;` (기존 8px에서 이동)
- **디자인:** 배경색 `#007bff`, 테두리 반경 `10px`, 그림자 추가, 흰색 테두리(`border: 2px solid rgba(255,255,255,0.2)`)를 추가하여 배경과 분리합니다.

### 2.3 메인 콘텐츠 레이아웃 보호 (Safety Margin)
- 사이드바가 접혔을 때 메인 콘텐츠(`stMainBlockContainer`) 좌측에 60px의 여백을 추가하여 버튼과 PDF 컨트롤바가 겹치지 않게 합니다.
- `is_expanded` 플래그를 활용하여 조건부 CSS를 주입합니다.

### 2.4 JavaScript를 통한 DOM 안정화
- Streamlit의 React 렌더링 주기에 상관없이 버튼을 항상 `document.body` 최상단에 유지하도록 MutationObserver 로직을 강화합니다.

## 3. 구성 요소별 변경 사항

### 3.1 `src/ui/ui.py`
- `inject_custom_css` 함수에서 `is_expanded` 값에 따라 `stMainBlockContainer`의 `padding-left`를 동적으로 변경.
- CSS 스타일 가이드 업데이트 (선택자 수정 및 신규 스타일 추가).
- JavaScript `fixSidebarBtn` 함수에서 최신 Streamlit ID 대응 로직 강화.

### 3.2 `src/main.py`
- 사이드바 상태를 판단하는 `is_expanded` 로직이 정확하게 `inject_custom_css`에 전달되는지 확인.

## 4. 테스트 및 검증 계획 (Verification)
- **사이드바 확장 시:** 좌측 여백이 15px(기본값)로 유지되는지 확인.
- **사이드바 축소 시:** 
    - 확장 버튼이 (15px, 15px) 위치에 정상 노출되는지 확인.
    - 메인 콘텐츠 좌측에 60px 여백이 생겨 PDF 컨트롤바와 버튼이 겹치지 않는지 확인.
- **브라우저 호환성:** Chrome, Edge 등 주요 브라우저에서 버튼 클릭 시 사이드바가 정상적으로 열리는지 확인.
