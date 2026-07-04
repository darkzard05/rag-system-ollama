# independent-scrolling - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->

**What you'll get:** PDF 뷰어(좌측)와 채팅창(우측)이 각각 독립적으로 스크롤되는 Streamlit 앱. 전체 페이지 스크롤바는 사라지고 각 컬럼 내부에서만 스크롤이 동작합니다. 채팅 입력창은 우측 컬럼 하단에 고정됩니다.

**Why this approach:** 6번의 다른 접근법이 모두 실패했습니다. 이번에는 Streamlit이 이미 생성하는 `st.container(height=N)`의 네이티브 `overflow-y: auto`를 활용하되, `height`를 덮어쓰는 대신 `height: auto + max-height: calc()`로 제한합니다. JS, flex chain, DOM 조작이 전혀 필요 없으며, 이미 존재하는 `[data-testid="stVerticalBlockBorderWrapper"] > div` 선택자(mobile CSS에서 검증됨)를 사용합니다.

**What it will NOT do:** 
- viewer.py, chat.py의 st.container() 호출 자체를 수정하지 않음
- streamlit_js_eval이나 JavaScript 사용하지 않음
- 수동 `<div>` 래핑(unsafe_allow_html) 사용하지 않음
- flex-chain CSS 속성 추가하지 않음
- 모바일에서 columns가 이미 stack되는 방식 변경하지 않음

**Effort:** Medium (4 files, 5-8 todos)
**Risk:** Medium - 이전 6회 실패 이력이 있으며, Streamlit DOM 구조 변경 가능성 존재
**Decisions to sanity-check:** 단일 `--viewport-offset: 280px` 값을 좌/우 컬럼 모두에 사용; chat input fixed positioning 추가

> TL;DR (machine): Medium effort, targeting layout.py + variables.py + ui.py CSS changes using `[data-testid="stVerticalBlockBorderWrapper"] > div` selector with `height: auto; max-height: calc(100dvh - offset)` strategy.

---

## Scope
### Must have
- C1: 브라우저 뷰포트 스크롤 잠금 (.stApp, .block-container overflow: hidden)
- C2: 좌측 PDF 컬럼 독립 스크롤 (stVerticalBlockBorderWrapper 내부 scroll div에 max-height 적용)
- C3: 우측 채팅 컬럼 독립 스크롤 (tabs 내부 포함, stVerticalBlockBorderWrapper 내부 scroll div에 max-height 적용)
- C4: 채팅 입력창 우측 컬럼 하단 고정 (stChatInputContainer fixed positioning)
- C5: 모바일 반응형 (768px 이하에서 max-height 제거, 기본 스크롤로 복원)
- C6: Playwright 기반 자동 검증

### Must NOT have (guardrails, anti-slop, scope boundaries)
- st.container 호출 자체를 제거/수정 금지 (viewer.py, chat.py 건드리지 않음)
- streamlit_js_eval/JavaScript 사용 금지
- 수동 `<div>` 래핑(unsafe_allow_html) 사용 금지
- flex-chain CSS (.stApp, .block-container 등에 display:flex) 추가 금지
- DOM 구조 변경 금지

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: **tests-after** — Playwright E2E tests verify independent scrolling behavior
- Evidence path: `.omo/evidence/task-*-independent-scrolling.*`
- Framework: Playwright (async) — already established in `scripts/verify_ui_scrolling.py` and `tests/e2e/test_chat_scroll.py`

## Execution strategy
### Parallel execution waves

**Wave 1 (Tasks 1-3)** — CSS foundation: variables + layout CSS + viewport locking → can be done in parallel as they modify different files
**Wave 2 (Tasks 4-5)** — Chat input + mobile media queries → depends on Wave 1 CSS structure
**Wave 3 (Task 6)** — Verification → depends on all CSS changes being in place
**Wave 4 (Tasks 7-8)** — Final verification wave + commit → after all changes verified

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1. CSS 변수 업데이트 | — | 4,5 | 2,3 |
| 2. Container max-height CSS | — | 4,5,6 | 1,3 |
| 3. Viewport locking 강화 | — | 6 | 1,2 |
| 4. Chat input 포지셔닝 | 1,2 | 6 | 5 |
| 5. Mobile 반응형 업데이트 | 1,2 | 6 | 4 |
| 6. Playwright 검증 | 2,3,4,5 | 7,8 | — |
| 7. 검증 스크립트 업데이트 | 6 | 8 | — |
| 8. 최종 확인 및 커밋 | 7 | — | — |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

### Wave 1: CSS Foundation

- [x] 1. `src/ui/styles/variables.py`: CSS 변수 업데이트
  What to do / Must NOT do:
  - `--viewport-offset` 값 검증: 현재 280px 유지 (좌측: header 55 + PDF nav 50 + buffer 175 = 280; 우측: header 55 + tabs 45 + chat input 80 + buffer 100 = 280)
  - `100dvh`에 대한 fallback으로 `100vh` 사용 (CSS cascade: `max-height: calc(100vh - var(--viewport-offset)); max-height: calc(100dvh - var(--viewport-offset))`는 variables.py가 아닌 layout.css에서 처리)
  - variables.py는 순수 CSS 변수 정의만 유지, 계산 로직은 layout.py에서 처리
  Parallelization: Wave 1 | Blocked by: — | Blocks: 4,5
  References: `src/ui/styles/variables.py:1-14` (current layout 변수)
  Acceptance criteria (agent-executable): `ruff check src/ui/styles/variables.py` passes
  QA scenarios:
  - Happy: `ruff format src/ui/styles/variables.py` → no errors
  - Failure: 의도치 않은 값 변경 확인 → `git diff`로 변경사항 검토
  Commit: Y | `refactor(ui): update CSS variables for independent scrolling layout`

- [x] 2. `src/ui/styles/layout.py`: Container max-height CSS 규칙 추가
  What to do / Must NOT do:
  - 기존 CSS 섹션 13 (lines 66-70)을 **완전히 교체**
  - 새 CSS: `.block-container [data-testid="stVerticalBlockBorderWrapper"] > div { height: auto !important; max-height: calc(100vh - var(--viewport-offset)); max-height: calc(100dvh - var(--viewport-offset)); overflow-y: auto !important; }`
  - 기존 `div[style*="height: 1000px"]` 선택자 제거 (취약한 선택자)
  - `[data-testid="stVerticalBlockBorderWrapper"] > div` 선택자 사용 (mobile media query에서 이미 검증된 선택자)
  - `height: auto !important`로 Streamlit의 inline `height: 1000px` 덮어쓰기
  - `overflow-y: auto !important`로 스크롤 유지 (Streamlit이 native로 설정하지만 명시적 선언)
  - **Must NOT**: flex-chain CSS 추가 금지, display 속성 변경 금지
  Parallelization: Wave 1 | Blocked by: — | Blocks: 4,5,6
  References: `src/ui/styles/layout.py:66-70` (current broken CSS), `layout.py:56-58` (mobile에서 `[data-testid="stVerticalBlockBorderWrapper"] > div` 사용 확인), `src/ui/styles/variables.py:1-14` (--viewport-offset)
  Acceptance criteria (agent-executable): 1) `ruff check src/ui/styles/layout.py` passes. 2) 드래프트 검토: `div[style*="height: 1000px"]` 선택자가 완전히 제거되었는지 확인
  QA scenarios:
  - Happy: `ruff format && ruff check .` passes
  - Failure: git grep `div[style *= "height: 1000px"]` 또는 `div[style*="height: 1000px"]` 반환값이 0인지 확인
  Commit: Y | `fix(ui): replace fragile container height override with auto+max-height strategy`

- [x] 3. `src/ui/styles/layout.py`: Viewport locking 강화
  What to do / Must NOT do:
  - 현재 `.block-container`에 `overflow: hidden !important` (line 7)는 유지
  - `.stApp`과 `.stAppViewContainer`에 `overflow: hidden` 추가 (현재 없는 경우에만)
  - 이미 layout.py에 있는지 확인: `grep "stApp.*overflow" src/ui/styles/layout.py`
  - **Must NOT**: `display: flex` 추가 금지, height 속성 변경 금지
  Parallelization: Wave 1 | Blocked by: — | Blocks: 6
  References: `src/ui/styles/layout.py:3-8` (current block-container), Streamlit 1.54.0 DOM: `.stAppV`가 메인 뷰포트 컨테이너
  Acceptance criteria (agent-executable): `ruff check src/ui/styles/layout.py` passes
  QA scenarios: 이후 Wave 3에서 Playwright로 검증 (body scrollbar 사라짐 확인)
  Commit: N (Task 2와 함께 커밋)

- [x] 4. `src/ui/styles/layout.py`: Chat input 포지셔닝 CSS 추가
  What to do / Must NOT do:
  - **Desktop**: `[data-testid="stChatInputContainer"]`에 `position: fixed !important; bottom: 0; right: 0; left: 50%; width: 50%` CSS 추가
  - 배경색: `var(--background-color)` 사용
  - 상단 테두리: subtle border-top
  - z-index: 100 (scroll container 위에 표시되도록)
  - 우측 컬럼 하단에 고정 (좌측 50%는 PDF 컬럼이므로 chat input은 우측 50%만 차지)
  - **Must NOT**: chat.py 파일 수정 금지, DOM 구조 변경 금지
  Parallelization: Wave 2 | Blocked by: 1,2 | Blocks: 6
  References: `plans/native-scroll-fix.md:43-52` (previous planning for chat input), `chat.py:495` (st.chat_input 호출 위치)
  Acceptance criteria: 1) `ruff check` passes. 2) CSS가 `[data-testid="stChatInputContainer"]`를 정확히 타겟팅
  QA scenarios: Playwright로 chat input 위치 검증 (left >= viewport_width * 0.45, bottom == 0)
  Commit: Y | `fix(ui): scope chat input to right column with fixed positioning`

- [x] 5. `src/ui/styles/layout.py`: Mobile 반응형 미디어 쿼리 업데이트
  What to do / Must NOT do:
  - 기존 mobile 섹션(lines 49-64) 유지 + `max-height` 제거 추가
  - 추가: `@media (max-width: 768px) { [data-testid="stVerticalBlockBorderWrapper"] > div { max-height: none !important; } }`
  - 기존 `height: auto` 설정은 유지 (mobile에서 이미 작동 중)
  - `st.chat_input`의 mobile `left: 0; width: 100%`은 이미 기존 CSS에 존재 (유지)
  Parallelization: Wave 2 | Blocked by: 1,2 | Blocks: 6
  References: `src/ui/styles/layout.py:49-64` (existing mobile CSS)
  Acceptance criteria: `ruff check` passes
  QA scenarios: 이후 Playwright로 768px 이하 viewport에서 max-height가 제거되었는지 확인
  Commit: N (Task 4와 함께 커밋)

### Wave 3: Verification

- [x] 6. Playwright 검증: 독립 스크롤 동작 확인
  What to do / Must NOT do:
  - `scripts/verify_ui_scrolling.py` 실행 (이미 존재)
  - `tests/e2e/test_chat_scroll.py` 실행 (이미 존재)
  - 결과를 `.omo/evidence/`에 저장
  - **필요시** verify_ui_scrolling.py 업데이트 (선택자를 새 CSS 구조에 맞게 조정)
  - **Must NOT**: 테스트 없이 CSS 변경사항 병합 금지
  Parallelization: Wave 3 | Blocked by: 2,3,4,5 | Blocks: 7
  References: `scripts/verify_ui_scrolling.py` (existing test), `tests/e2e/test_chat_scroll.py` (existing test)
  Acceptance criteria:
  1. `python scripts/verify_ui_scrolling.py` → exit code 0
  2. `python -m pytest tests/e2e/test_chat_scroll.py -v` → PASS
  3. Playwright 출력에서 "PASS" 또는 "SUCCESS" 확인
  QA scenarios:
  - Happy: PDF 컨테이너 스크롤 시 chat 컨테이너 scrollTop == 0 (독립성 확인)
  - Happy: Chat 컨테이너 스크롤 시 PDF 컨테이너 scrollTop == 0 (독립성 확인)
  - Happy: body/window scrollTop == 0 (viewport 잠금 확인)
  - Failure: 스크롤이 동기화되면 FAIL (한쪽 스크롤이 다른 쪽에 영향)
  Evidence: `.omo/evidence/task-6-verify-scroll.txt`
  Commit: N (검증만 수행)

- [x] 7. 검증 스크립트 업데이트 (필요시)
  What to do / Must NOT do:
  - verify_ui_scrolling.py의 CSS 선택자가 새 구조와 일치하는지 확인
  - `[data-testid="stVerticalBlockBorderWrapper"]`를 타겟으로 업데이트 (필요시)
  - 기존 테스트 로직 (overflowY, height, parentTestId 검사) 유지
  - **Must NOT**: 테스트 없이 skip
  Parallelization: Wave 3 | Blocked by: 6 | Blocks: 8
  References: `scripts/verify_ui_scrolling.py:37-84` (existing scrolling verification logic)
  Acceptance criteria: `ruff check scripts/verify_ui_scrolling.py` passes
  QA scenarios: `python scripts/verify_ui_scrolling.py` → exit code 0
  Commit: Y | `test(ui): update scrolling verification selectors for new CSS strategy`

### Wave 4: Final Verification

- [x] 8. Final verification: 모든 검증 통과 및 커밋
  What to do:
  1. `ruff check .` — zero errors
  2. `ruff format .` — consistent formatting
  3. Playwright 검증 재실행 — 독립 스크롤 확인
  4. 모든 변경사항 스테이징 및 커밋
  Parallelization: Wave 4 | Blocked by: 7 | Blocks: —
  Acceptance criteria: 모든 ruff 검사 통과, Playwright exit code 0
  Commit: Y | `feat(ui): implement independent column scrolling via auto+max-height strategy`

## Final verification wave (parallel, ALL must pass)
- [x] F1. Plan compliance audit — 모든 todo가 완료되었고 scope IN/OFF 준수 ✅
- [x] F2. Code quality review — ruff check, ruff format 통과 ✅
- [x] F3. Playwright E2E 검증 — 독립 스크롤 동작 확인 ✅
- [x] F4. Scope fidelity — viewer.py, chat.py 변경되지 않음 확인 (git diff --stat) ✅
- [x] F5. 이전 실패 접근법과 차별화 확인 — `div[style*="height: 1000px"]` 선택자 완전 제거 확인 ✅

## Commit strategy
1. **Task 2+3**: `fix(ui): replace fragile container height override with auto+max-height strategy`
2. **Task 4+5**: `fix(ui): scope chat input to right column with fixed positioning`
3. **Task 7** (if needed): `test(ui): update scrolling verification selectors for new CSS strategy`
4. **Final**: `feat(ui): implement independent column scrolling via auto+max-height strategy`

각 커밋은 atomic하며 독립적으로 revert 가능.

## Success criteria
- [ ] 브라우저 수준의 수직 스크롤바가 완전히 제거됨
- [ ] PDF 컬럼(좌측)이 독립적으로 스크롤됨 (채팅 컬럼 scrollTop == 0 유지)
- [ ] 채팅 컬럼(우측)이 독립적으로 스크롤됨 (PDF 컬럼 scrollTop == 0 유지)
- [ ] 채팅 입력창이 우측 컬럼 하단에 고정되어 표시됨
- [ ] 768px 이하(모바일)에서 기본 스크롤 동작으로 복원됨
- [ ] `ruff check .` zero errors
- [ ] 모든 Playwright E2E 테스트 PASS
- [ ] viewer.py, chat.py 수정 없음
- [ ] streamlit_js_eval, JavaScript 미사용
