# Project Rules (룰북)

실무에서 발견된, 문서화되지 않았지만 반드시 지켜야 할 제약/패턴. 위반 시 앱이
조용히 죽거나 미묘한 UI 버그가 발생한다.

## Streamlit

### R1. `st.iframe` height 는 0 불가
- `st.iframe(html, height=0)` → `StreamlitInvalidHeightError: Invalid height value: 0`
  (Height must be positive integer | "content" | "stretch").
- 0px로 숨기려면 `height="content"` 를 쓴다. `<script>` only 콘텐츠는 content 높이가
  ~0px 가까이 측정되어 사실상 보이지 않는다 (기존 `inject_header_height_script` 패턴).
- 위반 시 앱 전체가 런타임 에러로 죽고 화면이 안 뜬다.

### R2. CSS 주입은 본문 `<style>` 마크업 금지 — `<head>` 영구 주입 사용
- `st.markdown(f"<style>{css}</style>")` 방식은 upload `on_change` 풀 rerun 시
  레이아웃 delta 가 style delta 보다 먼저 플러시되어, flex:1 규칙이 늦게 적용되고
  그 사이 2열 컨테이너가 콘텐츠 높이로 붕괴하는 깜빡임이 발생한다
  (실증: 업로드 직후 840→420px 붕괴, 두 열 동일 축소).
- 해결: `st.iframe`(height="content") 안에서 JS 로 `window.parent.document.head` 에
  `<style id="...">` 를 append 하여 세션 내 재런에서도 스타일이 유지되게 한다
  (`src/ui/ui.py inject_custom_css` 참조).

### R3. `window.parent` 접근 필요 시 `st.html` 금지 → `st.iframe` 사용
- `st.html` 은 sandbox 가 `window.parent.document` 접근을 막는다.
- 부모 페이지(document) 에 접근해야 하는 스크립트(헤더 높이 감지, head CSS 주입 등)는
  `st.iframe` 을 써야 한다 (`src/ui/ui.py inject_header_height_script` 참조).
- `st.iframe` 안 스크립트는 `try/catch` 로 감싸 조용히 실패하게 한다.

## 검증

### V1. UI 타이밍 버그는 코드만으론 불충분 — 실제 DOM 캡처로 증명
- "깜빡임/레이아웃 붕괴" 류는 추론만으로 확정 못 함. Playwright 로 업로드 전/직후/후
  3시점의 컨테이너 높이(`stLayoutWrapper`, `stHorizontalBlock`, `stColumn`)를 캡처해
  수치로 증명한다 (`scripts/capture_flash.py` 패턴).
- 앱은 detached(`Start-Process`)로 띄우고, 수정 후 반드시 프로세스 재기동 후 재캡처.

### V2. 수정 후 곧바로 검증 단계로 (중간 멈춤 금지)
- 코드 수정 직후 LSP/ruff/mypy 와 실제 동작 캡처를 연속 실행한다. 사용자가
  "멈추지 말고 다음으로 넘어가라" 했으므로, 대기성 확인은 스킵하고 검증까지 밀어붙인다.
