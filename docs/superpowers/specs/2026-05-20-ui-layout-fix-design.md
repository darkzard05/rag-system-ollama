# UI Layout Overflow Fix Design

**날짜:** 2026-05-20
**상태:** Draft
**작성자:** Gemini CLI

## 1. 문제 정의
- **증상:** PDF 미리보기 컬럼과 채팅창 컬럼의 아랫부분이 화면(뷰포트)을 넘어가서 잘림.
- **원인:** 
    1. `src/ui/components/viewer.py` 및 `src/common/config.py`에서 `st.container(height=...)`에 800px, 700px 등 고정된 픽셀 값이 설정되어 있음.
    2. `src/ui/ui.py`에서 전체 화면을 `100dvh`로 고정하고 `overflow: hidden`을 설정하여, 내부 컨테이너가 뷰포트를 초과할 경우 스크롤 없이 잘리게 됨.

## 2. 해결 방안 (접근 방식)
### 접근 방식: CSS 기반 동적 레이아웃 최적화 (권장)
- **핵심 아이디어:** Streamlit 컨테이너의 하드코딩된 높이 스타일을 CSS `!important`로 덮어쓰고, Flexbox를 사용하여 가용한 화면 높이를 동적으로 채우도록 함.

#### 상세 변경 사항:
1. **전역 스타일 수정 (`src/ui/ui.py`):**
    - `st.container(height=...)`가 생성하는 `stVerticalBlockBorderWrapper`의 높이를 고정값이 아닌 부모 컨테이너의 100%를 차지하도록 강제함.
    - `calc(100dvh - [상단바+여백])`을 활용하여 메인 블록의 높이를 정확히 산출함.
2. **구성 요소 수정 (`src/ui/components/viewer.py`, `src/ui/components/chat.py`):**
    - `st.container` 호출 시 전달하는 `height` 값을 상징적인 값(예: 100)으로 낮추거나, CSS 타겟팅을 위해 유지하되 스타일로 제어함.
3. **설정값 조정 (`src/common/config.py`):**
    - `UI_CONTAINER_HEIGHT`의 기본값을 제거하거나 동적 처리를 위한 가이드로 변경.

## 3. 기대 효과
- 브라우저 창 크기에 상관없이 PDF 뷰어와 채팅 영역이 화면 하단까지 꽉 차게 표시됨.
- 하단 채팅 입력창이 항상 가시 영역 내에 유지됨.
- 내용이 길어질 경우에만 각 컬럼 내부에서 독립적인 스크롤이 발생함.

## 4. 검증 계획
- **수동 검증:** 브라우저 창 크기를 조절하며 하단 잘림 현상이 발생하는지 확인.
- **레이아웃 확인:** 개발자 도구(F12)를 통해 `[data-testid="stColumn"]`의 실제 계산된 높이가 뷰포트를 넘지 않는지 확인.
