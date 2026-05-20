# Adjust Chat Interface Container Height Design

**날짜:** 2026-05-21
**상태:** Approved
**작성자:** Gemini CLI

## 1. 문제 정의
- **상황:** Task 1에서 동적 높이를 위한 글로벌 CSS가 적용되었고, Task 2에서 PDF 뷰어 높이가 조정됨.
- **목표:** 채팅 인터페이스의 컨테이너 높이를 500으로 조정하여 "상징적 높이(symbolic height)"를 통일하고, CSS가 로드되기 전 레이아웃이 밀리는 현상을 방지함.

## 2. 해결 방안
- `src/common/config.py`의 `UI_CONTAINER_HEIGHT` 기본값을 700에서 500으로 변경.
- `src/ui/components/chat.py`에서 `render_chat_interface` 함수가 이 변수를 사용하는지 재확인.

## 3. 상세 변경 사항
### 3.1 `src/common/config.py`
```python
# AS-IS
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 700)

# TO-BE
UI_CONTAINER_HEIGHT: int = _ui_config.get("container_height", 500)
```

### 3.2 `src/ui/components/chat.py`
- `st.container(height=UI_CONTAINER_HEIGHT, border=False)` 부분 확인 및 유지.

## 4. 기대 효과
- 채팅창과 PDF 뷰어의 기본 높이가 500px로 일치됨.
- CSS의 `100%` 높이 오버라이드가 적용되기 전에도 안정적인 초기 레이아웃 제공.

## 5. 검증 계획
- **정적 분석:** `src/ui/components/chat.py`에서 `UI_CONTAINER_HEIGHT`가 올바르게 참조되는지 확인.
- **실행 확인:** Streamlit 앱 실행 시 채팅 컨테이너가 지정된 높이(또는 CSS 오버라이드된 높이)로 표시되는지 확인.
