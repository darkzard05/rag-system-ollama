# Design Spec: UI Layout Optimization (Responsive App-Shell)

- **Date**: 2026-06-08
- **Topic**: UI 레이아웃 높이 최적화 (채팅 및 미리보기창 화면 초과 이슈 해결)
- **Status**: Approved

## 1. Problem Statement
현재 채팅창과 PDF 뷰어의 높이가 고정값(`700px`)으로 설정되어 있어, 브라우저 해상도에 따라 화면을 초과하거나 공간이 낭비되는 문제가 발생함. 또한 전체 페이지 스크롤과 내부 컨테이너 스크롤이 중첩되어 UX가 저하됨.

## 2. Goals & Success Criteria
- [ ] 채팅창과 PDF 뷰어가 브라우저 뷰포트 높이(`100dvh`)를 넘지 않도록 고정.
- [ ] 각 영역이 독립적인 스크롤바를 가짐 (Independent Scrolling).
- [ ] 해상도와 기기에 상관없이 일관된 레이아웃 유지 (Responsive).
- [ ] `constants.py`의 하드코딩된 높이 의존성 제거.

## 3. Proposed Approach: Hybrid Offset (calc + dvh)
브라우저의 동적 뷰포트 높이(`dvh`)에서 상/하단 고정 요소의 높이를 뺀 값을 컨테이너 높이로 설정하는 방식.

### 3.1 architecture & Layout Control
- **Global CSS (`src/ui/ui.py`)**:
    - `.stApp`에 `overflow: hidden` 및 `height: 100dvh` 적용하여 전체 화면 스크롤 차단.
    - `.block-container` 패딩 최적화로 가용 면적 극대화.
- **Component Styling**:
    - 채팅창 및 PDF 뷰어 외곽 컨테이너에 전용 클래스 부여.
    - CSS `calc()`를 사용하여 높이 동적 계산.

### 3.2 Key Specifications
- **Chat Interface**: `height: calc(100dvh - 12rem);` (상단바 + 하단 입력창 + 여백 고려)
- **PDF Viewer**: `height: calc(100dvh - 10rem);` (상단 컨트롤바 + 여백 고려)
- **Fallback**: `dvh` 미지원 브라우저를 위해 `vh` 기반 폴백 제공.
- **Min-Height**: 최소 가독성 보장을 위해 `min-height: 400px` 설정.

## 4. Implementation Plan (Summary)
1. `src/ui/ui.py`: 전체 화면 고정 및 전역 레이아웃 CSS 주입.
2. `src/ui/components/chat.py`: 채팅 컨테이너에 클래스 부여 및 높이 제어 CSS 적용.
3. `src/ui/components/viewer.py`: PDF 뷰어 컨테이너에 클래스 부여 및 높이 제어 CSS 적용.
4. `src/common/constants.py`: 더 이상 사용되지 않는 `CONTAINER_HEIGHT` 상수 정리 (또는 기본값으로 유지).

## 5. Testing Strategy
- 다양한 브라우저 해상도(데스크탑, 태블릿 환경 시뮬레이션)에서 화면 초과 여부 확인.
- 채팅 메시지 누적 시 내부 스크롤 정상 작동 확인.
- PDF 페이지 전환 시 레이아웃 유지 확인.
