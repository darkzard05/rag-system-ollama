# 설계 문서: 메인페이지 레이아웃 및 독립 스크롤 리팩토링

- **날짜:** 2026-05-21
- **상태:** 초안 (사용자 승인 대기 중)
- **작성자:** Gemini CLI (Senior Software Engineer)

## 1. 개요 (Overview)
현재 RAG 시스템의 UI는 PDF 뷰어와 채팅창 컬럼이 병렬로 배치되어 있으나, 내용이 길어질 경우 화면 높이를 초과하여 전체 페이지에 스크롤이 발생합니다. 본 리팩토링은 각 컬럼을 뷰포트 높이(Viewport Height)에 고정하고, 내부에서 독립적으로 스크롤되도록 개선하여 사용자 경험(UX)을 최적화하는 것을 목표로 합니다.

## 2. 목표 (Goals)
- PDF 뷰어와 채팅창이 화면 높이(100dvh)를 넘지 않도록 고정.
- 두 영역이 각각 독립적으로 스크롤되도록 구현.
- 브라우저 해상도 및 모바일 환경에 대응하는 동적 높이 계산 적용.
- 하드코딩된 매직 넘버(예: 701px)를 제거하고 CSS 변수 기반의 체계적 관리 도입.

## 3. 상세 설계 (Detailed Design)

### 3.1 CSS 아키텍처 (CSS Variables)
`src/ui/ui.py`의 `inject_custom_css`에서 다음과 같은 CSS 변수를 정의합니다.

```css
:root {
    --app-padding-top: 2rem;       /* Streamlit 상단 기본 여백 보정 */
    --footer-height: 100px;        /* 채팅 입력창 영역 높이 */
    --pdf-controls-height: 60px;   /* PDF 페이지 네비게이션 높이 */
    --content-height: calc(100dvh - var(--app-padding-top) - 2rem);
}
```

### 3.2 핵심 스타일링 전략
1. **App Shell 고정:** `.stApp` 컨테이너에 `overflow: hidden`을 적용하여 페이지 전체 스크롤을 차단합니다.
2. **독립 스크롤 컨테이너:**
    - **PDF 영역:** 컨트롤바 아래의 컨테이너를 `height: calc(var(--content-height) - var(--pdf-controls-height))`로 설정.
    - **채팅 영역:** 메시지 목록 컨테이너를 `height: calc(var(--content-height) - var(--footer-height))`로 설정.
3. **선택자 최적화:** `div[data-testid="stVerticalBlock"]` 내의 컨테이너를 타겟팅하거나, Streamlit의 `st.container(height=...)`가 생성하는 고유 속성을 활용하여 안정적인 선택자를 사용합니다.

### 3.3 컴포넌트 변경 사항
- **`src/ui/components/viewer.py`**: `st.container(height=...)` 호출 시 인자값을 CSS 변수와 동기화하거나, 고유 키를 부여하여 CSS에서 정밀 제어할 수 있도록 수정.
- **`src/ui/components/chat.py`**: 채팅 메시지 컨테이너의 높이 설정을 뷰포트 기반으로 최적화.

## 4. 검증 계획 (Validation Plan)
- **시각적 검증:** 다양한 브라우저 창 크기에서 스크롤바가 각 컬럼 내부에만 생기는지 확인.
- **기능 검증:** 
    - PDF 페이지 이동 시 스크롤 위치 유지 여부 확인.
    - 채팅 메시지 추가 시 하단 고정 입력창이 밀려나지 않는지 확인.
- **회귀 테스트:** 기존의 PDF 하이라이트(Annotation) 및 채팅 스트리밍 기능이 정상 작동하는지 확인.

## 5. 자가 리뷰 (Self-Review)
- [x] **Placeholder scan:** 모든 수치 및 전략이 명시됨.
- [x] **Internal consistency:** CSS 변수와 컴포넌트 구조가 일치함.
- [x] **Scope check:** 메인 레이아웃 리팩토링에 집중됨.
- [x] **Ambiguity check:** 독립 스크롤 방식이 명확히 정의됨.
