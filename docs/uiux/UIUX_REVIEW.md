# GraphRAG-Ollama UI/UX 최종 리뷰 (Critical Review)

> 산출물: UIUX 리뷰 플랜 todo 9 (최종 리뷰 작성). 읽기 전용 리뷰로 product code 변경 없음.
> 일자: 2026-08-12
> 입력 체인: task-1-ui-inventory(84 인용) → 4개 레인 비평(lane-visual/usability/accessibility/interaction) → synthesis-notes(교차 분석·등급화) → task-2-design-intent(7 스펙·42 대조 포인트) → task-3-screens(화면 5종)
> 판정 원칙: 모든 평결은 (경로:행) 인용 또는 (screens/...) 화면 근거 기반. 무근거 주장 0건.
> 심각도 루브릭: P0=차단(사용 불가·데이터/진행 위험) · P1=주요(핵심 흐름 방해) · P2=개선 권장(마찰·비효율) · P3=미세(폴리시·일관성) (synthesis-notes.md:7)

## 1. 실행 요약

시각(A)·유용성(B, Nielsen 10)·접근성(C, WCAG 2.1 AA)·상호작용(D) 4개 레인의 독립 비평을 교차 분석해 34개 이슈로 통합했다. 등급 분포는 P0 0건, P1 4건, P2 20건, P3 10건이다 (synthesis-notes.md:95-97).

핵심 결론은 세 가지다. 첫째, 핵심 흐름을 방해하는 P1 4건이 모두 "사용자 통제" 계열이다. Reset All의 확인 절차 부재와 생성 중 활성(UX-1), 답변 완료 시 무단 자동 페이지 점프(INT-1), 취소·중단이 즉시 반영되지 않는 지연과 daemon 스레드 지속(INT-2), 모델 교체 시 메인 스레드 동기 로드로 인한 UI 전체 블로킹(INT-3)이다. 둘째, 시각·타이포는 토큰 체계 도입 초기 단계다. 스페이싱 토큰 5종은 정의됐으나 채택이 불일치하고(main.css:3-9,170,237), 브랜드 휴가 Streamlit 기본 레드에 종속되며(main.css:15-16), 타이포 계층이 크기(0.9/1/1.5/2rem)만으로 표현된다(lane-visual.md:9). 셋째, 접근성은 대비(추정 3.3:1 등), 포커스 링 부재, 줌 재배치 차단, 타깃 크기 미달에서 WCAG 2.1 AA 위반이 다수 확인됐다.

반면 유지할 강점도 명확하다. 업로드 즉시 3중 검증(main.py:418-421,425-429,447-461), 완료 답변의 참조 popover 페이지 버튼과 toast+즉시 rerun(chat.py:59-80, widget_keys.py:34), 스트리밍의 "준비 중..." placeholder와 ▌ 커서 연속 피드백(streaming.py:104-120), 컬럼 비율 42/58과 color-mix 파생 표면(main.css:169-171,295-300)이다. 이 7건은 개선 작업에서 퇴행 없이 보존해야 한다 (synthesis-notes.md:81-89).

## 2. 리뷰 방법론

### 2.1 렌즈 구성 (4레인)

- Lane A 시각 (lane-visual.md): 시각 계층·색·타이포·여백/스페이싱·일관성·브랜딩. 평결 14건(VIS-1~14, 인용 33).
- Lane B 유용성 (lane-usability.md): Nielsen 10 휴리스틱 전수 적용(N1~N10). 평결 13건(충족 3·미흡 10, 인용 49).
- Lane C 접근성 (lane-accessibility.md): WCAG 2.1 AA 기준. 평결 11건(실패 7·부분 충족 2·충족 1, 인용 26+screens 1+grep 2).
- Lane D 상호작용 (lane-interaction.md): 응답 대기·스트리밍 피드백·폴링 타임라인·취소/중단·점프·로딩·입력/스크롤. 평결 11건(긍정 1·비판 10, 인용 39).

### 2.2 평가 프레임워크와 근거 체계

- B는 Nielsen 10 휴리스틱, C는 WCAG 2.1 AA(1.4.3·1.4.4·1.4.10·2.1.1·2.3.3·2.4.7·2.5.5·2.5.8·3.3.1·3.3.2·1.3.1)를 평가 기준으로 삼았다.
- 근거는 task-1-ui-inventory.md(위젯·문구·상태·토큰·인터랙션 인벤토리), task-3-screens.md(화면 5종), 그리고 실제 코드 행 인용(경로:행)으로 구성했다. 모든 평결에 근거 인용을 포함했다.
- 4개 레인의 중복·중첩 지적은 교차 분석에서 통합 이슈로 병합했다. 통합 이슈 5건(UX-1·INT-1·INT-2·UX-2·UX-3·UX-4·UX-5)은 병합 출처를 "레이어 관점" 메타데이터로 보존하고, 단일 레인 지적은 그대로 승계했다 (synthesis-notes.md:11-24).
- 등급 부여는 위 루브릭을 적용했다. P0은 부여하지 않았다. 그 근거는 ① task-3 구동 확인(HTTP 200 + 유효 화면 5종)으로 "사용 불가" 사례가 없고, ② 세션 상태는 in-memory라 영구 데이터 손실이 없으며, ③ 최대 위험 동작인 Reset All은 사용자 발의·명시 라벨에 의한 세션 소거로 확인 절차 부재가 P1(핵심 흐름 방해) 범주에 해당하기 때문이다 (synthesis-notes.md:97).

### 2.3 한계

- 05-chat-message·06-streaming 화면은 미캡처다 (task-3-screens.md:22-24). 스트리밍 관련 판정은 코드 경로(streaming.py·chat.py) 기반이며, 화면 근거는 01-empty·02-mobile-390·03-dark·04-light·07-upload 5종으로 제한된다.
- 접근성 대비율은 코드 토큰 기반 산출로 전부 추정값이다 (lane-accessibility.md:5). 실제 픽셀 판독이 아니므로 절대값이 아닌 상대 비교로 해석해야 한다.
- 런타임 확인은 HTTP 200 + 유효 이미지 5종(task-3-screens.md:5,16-20)으로 구동 이상이 없음을 확인했으나, 모델 추론 품질·속도는 리뷰 범위 밖이다.

## 3. 영역별 리뷰

### 3.1 사이드바

관찰: 업로더·모델 셀렉터 2종·액션 버튼 3종이 한 컬럼에 밀집했다. 라벨은 영문("Upload PDF Document", "Reset All"), 한글("새 대화"), 이모지+한글("↻ 모델 새로고침")이 혼재하고 명명 규칙(명사구/동사구)도 제각각이다 (sidebar.py:73,88,106,117,141,153,166). 새로고침·새 대화·Reset All 3개 버튼이 전부 primary 풀폭 레드로 같은 시각 위계다 (sidebar.py:119,155,168).

근거: Reset All은 확인 절차 없이 즉시 reset_all_state()+st.rerun()으로 대화·파이프라인 상태를 소거하고, disabled 미지정이라 빌드·생성 중에도 활성 상태로 남는다 (sidebar.py:165-172,173). 모델 셀렉터는 원시 모델 ID("qwen3:4b-instruct-2507-q4_K_M")를 그대로 노출한다 (config.yml:7, sidebar.py:105-112). 핵심 개념 도움말은 빈 상태 가이드(config.yml:200)와 툴팁 3곳(sidebar.py:120,156,169)이 전부다.

판정: 파괴적 동작의 확인 절차 부재와 생성 중 활성은 P1(UX-1), 라벨 혼용과 시각 위계 파편화는 P2(UX-2·VIS-9), 원시 모델 ID 노출은 P2(UX-7), 도움말 공백은 P2(UX-8)다. 반면 업로드 즉시 3중 검증(main.py:418-421,425-429,447-461)과 해시 중복 빌드 차단(main.py:463)은 오류 예방 측면에서 유지 판정이다 (lane-usability.md:16).

### 3.2 PDF 뷰어

관찰: 네비게이션 컨트롤바는 뷰어 본문 아래에서 호출되어 하단 배치를 유지한다 (viewer.py: _display_pdf_viewer 호출 후 _display_pdf_controls 호출). 페이지는 뷰어 위젯 키에 포함돼 페이지 변경 시 재마운트된다 (widget_keys.py:39-44).

근거: 페이지 전환·컨트롤 클릭마다 뷰어 fragment가 재실행되고 스크롤이 상단으로 리셋되어 긴 문서의 열람 위치가 유실된다 (viewer.py:204-205,252-260, widget_keys.py:39-44). 답변 완료 시 첫 참조 페이지로 자동 점프하며, manual_nav_ts 가드는 토큰 생성 이후의 수동 이동만 보호하므로 생성 중 다른 페이지를 읽던 사용자가 강제 이동된다 (streaming.py:260-268, viewer.py:116-118,153-159). 참조 popover의 점프 버튼은 5컬럼 그리드로 390px 화면에서 개당 약 64×40px(추정)에 그쳐 44×44px AA 타깃에 미달한다 (chat.py:72-78, main.css:322-324, screens/02-mobile-390.png).

판정: 하단 네비게이션 배치는 설계 의도대로 유지 중이다. 반면 무단 자동 점프(INT-1, P1)와 열람 위치 유실(INT-6, P2)이 핵심 결함이다. 점프 자체의 효율(B/N7, toast+즉시 rerun)은 유지하되, 이동 결정권을 사용자에게 돌려야 한다 (lane-interaction.md:16).

### 3.3 채팅 타임라인

관찰: 타임라인은 @st.fragment(run_every=...)로 1초 폴링 전체 재렌더를 수행한다 (chat.py:275, config.yml:196). 업로드 후 RAG 빌드 진행은 st.status+st.progress+"% 완료"로 라이브 반영되고 완료 시 "✅ 분석 완료"로 전환된다 (chat.py:307-328,314).

근거: 문서(AGENTS.md:24)는 폴링을 "every 2s"로 기술하지만 실제 값은 1.0초다 (config.yml:196). 1초 주기 재렌더는 상단 참조를 읽던 사용자의 스크롤을 흔들 수 있다. 스트리밍 갱신은 백그라운드 스레드가 기록한 내용을 폴링이 읽는 구조라 최대 1 폴링 간격(~1s)의 지연이 발생한다 (streaming.py:104-120, chat.py:275).

판정: 빈 구간 없는 연속 피드백("준비 중..." placeholder·▌ 커서·단계 실시간 노출)은 유지할 강점이다 (chat.py:395-406, streaming.py:104-120). 문서·구현 주기 불일치(INT-5, P2)는 문서 갱신으로 즉시 해소 가능하다.

### 3.4 상태·피드백

관찰: 스트리밍은 st.status("🤔 {status_text}") expanded+▌ 커서로 진행을 보여주지만, 완료 시 status 박스가 통째로 사라지고 별도 캡션 "✅ 답변 생성 완료"로 보완한다 (chat.py:381-398,137-150, streaming.py:204). 완료 답변에는 완료 캡션·성능 메트릭·사고 과정 확장이 전부 기본 노출된다 (chat.py:145-150,173,193-223).

근거: 상태 전환의 단절·이중 표기(UX-6)와 완료 정보 3겹의 인지 부하(UX-3)가 확인됐다. 오류는 "답변 생성 중 오류가 발생했습니다."(streaming.py:32)로 원인 힌트가 없고, 재시도는 동일 질문 재타이핑뿐이다 (streaming.py:45-51, chat.py:372-375). 중단 클릭 시 버튼만 사라지고 상태 박스는 "생성 중"으로 잔존하며, 동기 LLM 구간에서는 다음 await까지 무반응이고 2초 join 초과 시 daemon 스레드가 백그라운드에서 RAG를 계속 실행한다 (streaming.py:278-282,152-154,362-371, chat.py:408-416). 빌드 취소도 체크포인트 도달 전까지 진행 표시가 그대로 남는다 (chat.py:329-336, main.py:293-303).

판정: 진행 피드백 자체는 충실하나 전환·취소·오류 복구 지점에서 단절된다. 취소·중단 지연은 P1(INT-2), 완료 정보 과다는 P2(UX-3), 상태 이중 표기는 P2(UX-6), 재시도 경로 부재는 P2(UX-4)다.

### 3.5 모바일·테마

관찰: 테마는 .streamlit/config.toml에 [theme] 섹션이 없어 OS 색상 체계를 자동 추종한다 (main.css:12,15-17,23). 다크/라이트 모두 브랜드 색이 Streamlit 기본 레드 #ff4b4b다 (main.css:15-16). 모바일 768px 이하에서는 두 컬럼이 50vh 고정 스택으로 전환된다 (main.css:326,342-347).

근거: 390px 화면에서 PDF·채팅이 각자 스크롤되는 "앱 두 개" 구조로 보인다 (main.css:342-347, screens/02-mobile-390.png). 100dvh 고정+overflow hidden은 200% 텍스트/줌 재배치를 차단하고, 모바일 50vh는 확대 시 패널 접근을 부분 차단한다 (main.css:30,90,342-347, ui.py:37). :focus-visible 전역 선언 0건, prefers-reduced-motion 가드 0건으로 키보드 탐색 가시성과 모션 선호가 무시된다 (main.css:248-251,313-319, grep 결과).

판정: 테마 자동 적응과 color-mix 파생 표면(main.css:169-171,295-300)은 유지할 강점이다. 모바일 주배치(VIS-4, P2), 브랜드 휴 부재(VIS-1, P2), 접근성 미달(ACC-1~ACC-9, P2~P3)은 개선 대상이다.

## 4. 심각도 등급 이슈 목록

등급순 정렬(P1 → P2 → P3). 근거는 합성(synthesis-notes.md)의 병합 출처를 기준으로 요약했다.

| ID | 등급 | 이슈 | 근거 | 권장 |
|---|---|---|---|---|
| [UX-1] | P1 | Reset All 파괴적 동작: 확인 절차 부재 + 빌드/생성 중 활성 | (sidebar.py:165-172,173 · main.css:16) | 확인 다이얼로그 또는 5초 되돌리기 토스트 + 생성 중 disabled + secondary 위계 |
| [INT-1] | P1 | 답변 완료 시 무단 자동 페이지 점프, 수동/자동 점프 반응 비대칭 | (streaming.py:260-268 · viewer.py:116-118,153-159 · chat.py:36-38) | "참조 페이지로 이동" 버튼/안내로 전환하거나 설정에서 끄기 + 점프 toast 유지 |
| [INT-2] | P1 | 취소·중단 즉시 반영 부재, daemon 스레드 지속, 상태 미확정 | (streaming.py:278-282,362-371 · chat.py:329-336,408-416 · main.py:293-303) | "중단 중..." 즉시 전환 + "중단됨" 확정 캡션 + rag_core abort 신호 연계 |
| [INT-3] | P1 | 모델 교체 시 메인 스레드 동기 load_llm, UI 전체 블로킹 + 진행 피드백 부재 | (main.py:646-649,377-383 · chat.py:351-353) | 백그라운드 워커 + st.status 진행 노출 |
| [UX-2] | P2 | 사이드바 라벨 영·한·이모지 혼용, 명명 규칙 불일치 | (sidebar.py:73,88,106,141,166 vs 117,153 · config.yml:200) | 한국어 통일 + 위젯 유형별 표기 규칙 고정 |
| [UX-3] | P2 | 완료 답변 정보 3겹·캡션 4연속 밀집, 인지 부하 과다 | (chat.py:145-150,173,193-223) | 메트릭·사고 과정 기본 접힘, 완료 상태만 노출 |
| [UX-4] | P2 | 오류 후 재시도 경로 부재 + 복구 제안 없는 문구 | (streaming.py:32,45-51 · chat.py:372-375) | st.error에 "재시도" 버튼(직전 쿼리 재전송) |
| [UX-5] | P2 | 빈 상태 가이드·placeholder 중복 노출, 상태 무관 유지 | (main.py:655-664 · chat.py:238-240,454-458 · config.yml:200) | 상태별 placeholder 분리 + 업로더 위치(사이드바) 안내 |
| [UX-6] | P2 | 스트리밍 완료 시 status 박스 소멸 + 캡션 이중 표기 | (chat.py:381-398,137-150 · streaming.py:204) | 완료 시 status를 state="complete"로 단일 전환 |
| [UX-7] | P2 | 모델 셀렉터에 원시 모델 ID 그대로 표시 | (config.yml:7 · sidebar.py:105-112) | 역할 라벨("4B 경량·빠름") 병기 |
| [UX-8] | P2 | 핵심 개념 도움말 부재(LLM/임베딩 구분, 문서 분석 단계 등) | (config.yml:200 · sidebar.py:120,156,169) | 인라인 "?" 도움말 또는 첫 실행 온보딩 |
| [VIS-1] | P2 | 브랜드 색이 Streamlit 기본 #ff4b4b 종속, 차별성 없음 | (main.css:15-16 · sidebar.py:23 · main.py:660) | 전용 휴 토큰 1개 + 파생 톤 |
| [VIS-2] | P2 | 스페이싱 토큰 채택 불일치(8px 하드코딩 2곳) | (main.css:5,170,237) | 하드코딩 8px을 var(--spacing-sm)으로 치환 |
| [VIS-3] | P2 | 타이포 체계 부재(font-family·@font-face 0건) | (main.css:1-398, font-family 0건) | 시스템 스택 + 한글 폴백 + letter-spacing 계층 |
| [VIS-4] | P2 | 모바일 PDF·채팅 50vh 고정 스택, "앱 두 개" 인상 | (main.css:342-347 · screens/02-mobile-390.png) | 35vh 컴팩트 프리뷰 + 채팅 지배열 |
| [ACC-1] | P2 | primary 레드 위 흰 텍스트 대비 추정 3.3:1(일반 텍스트 4.5:1 미달) | (main.css:16 · sidebar.py:152,165) | #c43c3c 계열로 어둡게 하거나 텍스트 다크 |
| [ACC-2] | P2 | 포커스 링 전역 부재(:focus-visible 0건) | (main.css:248-251 · grep :focus-visible 0건) | 전역 :focus-visible 2px·3:1 링 |
| [ACC-3] | P2 | 점프·Reset All·rerun 후 포커스 body 리셋 | (chat.py:38 · sidebar.py:173 · main.py:688) | 목표 앵커 복원 또는 aria-live 안내 |
| [ACC-4] | P2 | 100dvh 고정 + overflow hidden, 200% 줌 재배치 차단 | (main.css:30,90,342-347 · ui.py:37) | 줌 재배치 세로 스택 미디어 쿼리 + overflow 완화 |
| [ACC-5] | P2 | 참조 점프 버튼 5컬럼(390px 기준 약 64×40px), 44×44px 미달 | (chat.py:72-78 · main.css:322-324 · screens/02-mobile-390.png) | 3컬럼 이하 제한 또는 높이 44px 강제 |
| [ACC-6] | P2 | 인용 하이라이트 #007bff 대비 3.98:1(흰)/3.5:1(다크) | (utils.py:406-408) | weight 700 + #0056b3 또는 링크 파생 토큰 |
| [INT-4] | P2 | 생성 중 입력·업로더·셀렉터 전면 비활성, 수 분 대기 | (chat.py:449-458 · sidebar.py:77,111,158 · main.css:254-256) | 연속 질문 예약 또는 "완료 후 입력 가능" 안내 |
| [INT-5] | P2 | 폴링 1.0s vs 문서 "every 2s" 불일치, 재렌더 스크롤 흔들림 | (AGENTS.md:24 · chat.py:275 · config.yml:196) | 문서 갱신 + push/간격 축소 검토 |
| [INT-6] | P2 | 페이지 전환·컨트롤 클릭마다 스크롤 상단 리셋 | (widget_keys.py:39-44 · viewer.py:204-205,252-260) | 스크롤 복원 또는 컨트롤·뷰어 fragment 분리 |
| [VIS-5] | P3 | 아바타·섹션·버튼·캡션 이모지 6계열 혼재 | (chat.py:98,240,263,324 · sidebar.py:88,117) | 단일 아이콘 패밀리로 통일 |
| [VIS-6] | P3 | 접기 버튼 그림자 rgba(0,0,0,0.1) 하드코딩, 다크에서 소실 | (main.css:23,291) | color-mix 토큰화 |
| [VIS-7] | P3 | 빈 상태 스플래시가 사이드바 로고와 문구 3중 복제 | (sidebar.py:15-29 · main.py:655-664) | 스플래시 전용 시각 아이덴티티 |
| [VIS-8] | P3 | PDF|채팅 컬럼 gap 0.25rem 강제, 분리선 부재 | (main.css:89 · ui.py:93) | gap var(--spacing-md) 또는 1px 세로 구분선 |
| [UX-9] | P3 | RAG 빌드 단계명("분할→색인") 원문 노출 | (chat.py:307-328,275 · config.yml:196) | 일반 사용자 문구 매핑 |
| [ACC-7] | P3 | 로고 태그라인 0.9rem+opacity 0.8, 대비 추정 2.7~3.0:1 | (sidebar.py:23) | opacity 제거 + muted 변형 |
| [ACC-8] | P3 | 비활성 채팅 입력 opacity 0.6, 대비 추정 ~3.7:1 | (main.css:254-256 · chat.py:466) | opacity 0.85 이상 |
| [ACC-9] | P3 | msgFadeIn 0.3s, reduced-motion 가드 없음 | (main.css:313-319 · grep prefers-reduced-motion 0건) | @media (prefers-reduced-motion: reduce) 애니메이션 해제 |
| [ACC-10] | P3 | PDF 업로더 accept 실패 시 원인 안내 부재 | (sidebar.py:73 · chat.py:119) | 한 줄 안내 문구 추가 |
| [ACC-11] | P3 | sidebar.py:3-4 주석과 실제 코드(라벨 가시 표시) 불일치 | (sidebar.py:3-4,72-73,105-106 · viewer.py:291) | 주석 갱신 |

이슈 수 34건(요구 ≥ 10 충족), P0 0건(강제 부여 없음), 무근거 행 0건 (synthesis-notes.md:95-98). 이 목록에서 제외된 긍정·유지 판정 7건(컬럼 비율 42/58, color-mix 표면, msgFadeIn 절제, 업로드 3중 검증, 참조 popover, 스트리밍 연속 피드백, 라벨 접근성 충족)은 개선 시 퇴행하지 않도록 보존 대상이다 (synthesis-notes.md:81-89).

## 5. 우선순위 권장 개선사항

### 5.1 P1 긴급 처리 (4건)

1. Reset All 파괴 동작 안전장치 (UX-1)
   - 목표: 파괴적 동작에 확인 절차를 두고 생성 중 활성화를 막는다.
   - 방법: 클릭 시 확인 다이얼로그(또는 5초 되돌리기 토스트)와 빌드·생성 중 disabled, 시각 위계는 secondary로 하향한다 (sidebar.py:165-172,173 · main.css:16).
   - 기대 효과: 세션 소거 사고와 진행 작업 무효화를 원천 차단한다.

2. 자동 페이지 점프를 사용자 의지로 전환 (INT-1)
   - 목표: 답변 완료 시 강제 이동을 제거하고 사용자가 이동을 결정하게 한다.
   - 방법: 자동 점프 대신 "참조 페이지로 이동" 버튼/안내로 바꾸거나 설정에서 끄고, 이동 시 toast 안내는 유지한다 (streaming.py:260-268 · viewer.py:116-118,153-159 · chat.py:36-38).
   - 기대 효과: 생성 중 다른 페이지를 읽던 사용자의 화면이 도중에 바뀌지 않는다.

3. 취소·중단의 즉시 반영과 확정 피드백 (INT-2)
   - 목표: 중단 클릭이 즉시 UI에 반영되고 상태가 확정되며 백그라운드 실행이 종료된다.
   - 방법: 취소 접수 직후 "중단 중..." 상태 전환, 부분 답변에 "중단됨" 캡션, rag_core 레벨 abort 신호 연계, 빌드 취소도 "취소 요청됨"으로 즉시 전환한다 (streaming.py:278-282,362-371 · chat.py:329-336,408-416).
   - 기대 효과: 느린 로컬 모델에서도 수십 초 무반응 없이 사용자 통제가 체감된다.

4. 모델 교체 블로킹 제거 (INT-3)
   - 목표: 모델 교체 중에도 UI가 응답하고 진행이 보인다.
   - 방법: load_llm을 백그라운드 워커로 옮기고 st.status로 진행 상태를 노출한다 (main.py:646-649,377-383 · chat.py:351-353).
   - 기대 효과: 설정 흐름이 수 초~수 분 정지하지 않고 상태 로그가 즉시 렌더된다.

### 5.2 P2 고임팩트 (우선 실행 권장)

5. 라벨·명명 규칙 단일화 (UX-2): 사이드바 라벨을 한국어로 통일하고 위젯 유형별 표기 규칙을 고정한다 (sidebar.py:73,106,117,153,166). 영문은 보조 툴팁으로 내린다. 기대 효과: 마찰과 시각 리듬 파편화 해소.
6. 완료 정보 접힘 기본값 (UX-3): 메트릭·사고 과정 확장을 기본 접힘으로 내리고 완료 상태만 노출한다 (chat.py:145-150,173,193-223). 기대 효과: 반복 답변의 판독 비용 감소.
7. 브랜드 휴 도입 (VIS-1): 기본 #ff4b4b 대신 전용 휴 토큰 1개를 정의하고 파생 톤을 거기서 파생한다 (main.css:15-16). 기대 효과: 태그라인·포커스·컨트롤바 틴트의 차별성 확보.
8. 전역 포커스 링 + reduced-motion 가드 (ACC-2·ACC-9): :focus-visible 2px·3:1 링과 prefers-reduced-motion 애니메이션 해제를 추가한다 (main.css:248-251,313-319). 기대 효과: 키보드 탐색 가시성과 모션 선호 준수.
9. 모바일 주배치 개선 (VIS-4): 50vh 고정 스택을 35vh 컴팩트 프리뷰+채팅 지배열로 바꾼다 (main.css:342-347 · screens/02-mobile-390.png). 기대 효과: "앱 두 개" 인상 제거.
10. 타임라인 폴링 문서 정합 (INT-5): AGENTS.md의 "every 2s"를 실제 값(config.yml:196의 1.0s)으로 갱신하고 스크롤 흔들림 대응을 검토한다 (chat.py:275 · AGENTS.md:24). 기대 효과: 문서 신뢰와 재렌더 부담 완화.

## 6. 설계 의도 vs 구현 대조

대조 방법: task-2-design-intent.md가 추출한 7개 스펙·42개 대조 포인트를 현재 구현(main.css·viewer.py·chat.py·widget_keys.py 등)과 직접 대조해 충족/괴리/대체/회귀로 판정했다. 스펙 간 진화(05-21 CSS 변수 → 06-08 calc+dvh → 06-13 Flex/네이티브 분기)를 고려해, 후속 스펙으로 전략이 대체된 항목은 "대체"로 표기한다 (task-2-design-intent.md:108-112).

### 6.1 독립 컬럼 스크롤 (05-21 → 06-13 계열)

판정: 충족.
- .stApp에 100vh→100dvh 이중 선언+overflow hidden으로 브라우저 스크롤바를 차단한다 (main.css:30). 이는 뷰포트 락 의도(independent-scrolling-design.md:13-15, ui-layout-optimization-design.md:21-22, main-page-refactoring-design.md:31)와 일치한다.
- stColumn에 flex+min-height:0+overflow-y auto를 적용해 컬럼 내부 독립 스크롤을 트리거한다 (main.css:93-99). 06-13 스펙이 "핵심"으로 명시한 min-height:0(independent-scrolling-design.md:21,25-26)이 그대로 구현됐다.
- 브라우저 수직 스크롤바 없는 앱-쉘은 task-3 화면에서도 확인된다 (screens/01-empty.png).

### 6.2 chat input 스코핑 (06-13 두 스펙)

판정: 기능 충족, 구현 수단 괴리.
- 스펙은 [data-testid="stChatInputContainer"]에 width:50%+left:50%+background:transparent를 명시했다 (independent-scrolling-design.md:29-33, adaptive-native-layout-design.md:28-29).
- 실제 구현은 해당 셀렉터를 쓰지 않고, 채팅 컬럼의 last-child 래퍼를 margin-top:auto+sticky bottom으로 고정하는 방식이다 (main.css:223-234).
- 기능 목표(chat input이 채팅 컬럼 하단에만 존재, 전역 하단 바 아님)는 충족하므로 06-13 성공 기준(independent-scrolling-design.md:46-50)은 통과한다. 다만 "50% 중앙 스코프" 형태와 Streamlit 1.60 DOM 계약 기반 접근(adaptive-native-layout-design.md:19-21)은 다른 수단으로 대체됐다.

### 6.3 PDF 하단 네비게이션 (06-11)

판정: 충족.
- render_pdf_area에서 _display_pdf_viewer 호출 후 _display_pdf_controls를 호출해 컨트롤을 본문 아래에 배치한다. 06-11 스펙의 "컨테이너 아래로 이동"(pdf-nav-redesign.md:12-13)이 유지된다.
- 3컬럼(col_prev, col_page, col_next)으로 단순화됐다. 06-11 스펙의 4컬럼→3컬럼 전환(pdf-nav-redesign.md:18-19)이 적용됐다. 좌(문서 조작)·우(질의 입력) 하단 조작계 통일 의도(pdf-nav-redesign.md:27)도 실현됐다.

### 6.4 성능 지표 UI (06-10)

판정: 괴리.
- 스펙은 커스텀 HTML/CSS 데이터 테이블(.perf-table, 13px 폰트)과 ✨/🟢/⚠️ 상태 라벨, 임계값 표기를 명시했다 (performance-metrics-ui-improvement-design.md:11-13,20-50,57-60).
- 실제는 st.caption 1줄 메트릭으로 렌더된다 (chat.py:173). main.css에는 perf-table 규칙이 없고 "네이티브 컴포넌트(st.expander, st.status, st.caption)로 대체됨" 주석만 남았다 (main.css:1-398 주석).
- 정보 밀도 목표는 부분 충족이나, 스펙의 상태 피드백 아이콘과 임계값 기반 판정 표시는 미구현이다. 폰트 16~18px→13px 축소와 팝업 최적화 의도(performance-metrics-ui-improvement-design.md:11-12)도 커스텀 테이블 부재로 미달이다.

### 6.5 calc+dvh 하이브리드 (06-08 두 스펙)

판정: 대체.
- main.css에 calc(100dvh - Xrem) 공식은 0건이다. 06-08 스펙의 offset 기반 높이 계산(ui-layout-optimization-design.md:28-29, ui-height-optimization-design.md:15-19)은 06-13 Flex-Container 전략으로 대체됐다. 이는 task-2가 기록한 전략 분기(task-2-design-intent.md:109)와 일치한다.
- dvh 미지원 폴백은 유지된다. 100vh→100dvh 이중 선언(main.css:30)이 06-08 스펙의 vh 폴백 요구(ui-layout-optimization-design.md:30-31)를 충족한다.
- CONTAINER_HEIGHT 상수 의존성은 제거됐다. src 전체에서 상수 정의·참조 0건으로, 06-08 스펙의 "하드코딩 제거"(ui-layout-optimization-design.md:14,37) 목표를 달성했다.

### 6.6 모바일 폴백 (06-13)

판정: 부분 충족, 회귀 위험.
- 768px 미디어 쿼리에서 .stApp에 height:auto+overflow:visible을 재적용한다 (main.css:326-341). 이는 06-13 스펙의 "모바일에서 표준 스크롤 복귀"(independent-scrolling-design.md:43)를 따른다.
- 그러나 직후 컬럼에 50vh 고정+overflow-y:auto를 재적용한다 (main.css:342-347). 390px 화면에서 두 패널이 각자 스크롤되는 이중 앱 구조가 된다 (screens/02-mobile-390.png). 스펙이 의도한 단일 표준 스크롤과 다른 지점으로, VIS-4(모바일 주배치) 이슈와 연결된다.

### 6.7 페이지 이동 시 스크롤 위치 유지 (05-21 검증 항목)

판정: 회귀.
- 05-21 스펙의 검증 항목 "PDF 페이지 이동 시 스크롤 위치 유지"(main-page-refactoring-design.md:44-46)가 위반된다.
- 페이지를 뷰어 위젯 키에 포함한 재마운트(widget_keys.py:39-44)로 페이지 전환·컨트롤 클릭마다 스크롤이 상단으로 리셋된다 (viewer.py:204-205,252-260). INT-6(P2)과 동일 지점이다.
- 스크롤 복원 로직 또는 컨트롤·뷰어 fragment 분리가 필요하다.

## 7. 부록: 디자이너 레인 비평 원문

4개 레인의 평결 원문은 .omo/evidence/uiux-review/lane-*.md에 보존되어 있다. 본 부록은 각 레인의 구성과 핵심 판정 요약을 담는다.

- Lane A 시각: .omo/evidence/uiux-review/lane-visual.md (VIS-1~14, 평결 14건, 인용 33)
- Lane B 유용성: .omo/evidence/uiux-review/lane-usability.md (N1~N10, 평결 13건, 인용 49)
- Lane C 접근성: .omo/evidence/uiux-review/lane-accessibility.md (WCAG 2.1 AA, 평결 11건, 인용 26+screens 1+grep 2)
- Lane D 상호작용: .omo/evidence/uiux-review/lane-interaction.md (INT-1~11, 평결 11건, 인용 39)

### Lane A 시각 디자인 (lane-visual.md 요약)

- 핵심 지적: 브랜드 컬러 기본 레드 종속(VIS-1), 타이포 체계 부재(VIS-3), 스페이싱 토큰 채택 불일치(VIS-2), 모바일 50vh 이중 앱(VIS-14), 이모지 6계열 혼재(VIS-7), primary 3버튼 동일 위계(VIS-6).
- 유지 판정: 컬럼 비율 42/58과 헤더 JS 실측(VIS-10), color-mix 파생 표면(VIS-11), msgFadeIn 0.3s 단일 애니메이션(VIS-12).

### Lane B UX·유용성 (Nielsen 10) (lane-usability.md 요약)

- 충족 3건: 빌드 상태 가시성(N1), 업로드 3중 검증(N5), 참조 popover 페이지 버튼(N7).
- 미흡 10건: Reset All 확인 부재·자동 점프·취소 지연(N3), 라벨 혼용(N2·N4), 원시 모델 ID(N6), 완료 정보 과다(N8), 오류 복구 힌트 부재(N9), 핵심 개념 도움말 부재(N10), 상태 전환 이중 표기(N1).

### Lane C 접근성 (WCAG 2.1 AA) (lane-accessibility.md 요약)

- 실패 7건: primary 대비 3.3:1(1.4.3), 태그라인 대비(1.4.3), 비활성 입력 대비(1.4.3), 인용 하이라이트 대비(1.4.3), rerun 후 포커스 리셋(2.1.1), 포커스 링 부재(2.4.7), 줌 재배치 차단(1.4.4/1.4.10), reduced-motion 가드 부재(2.3.3), 타깃 크기 미달(2.5.8/2.5.5).
- 부분 충족 2건: 오류 식별(3.3.1/3.3.3, 업로더 accept 안내 부재), 라벨(1.3.1/3.3.2, 주석 불일치).
- 충족 1건: 업로더·셀렉트박스 라벨 가시 유지.
- 모든 대비율은 코드 토큰 기반 추정 (lane-accessibility.md:5).

### Lane D 상호작용·스트리밍 (lane-interaction.md 요약)

- 긍정 1건: 스트리밍 "준비 중..." placeholder+▌ 커서+단계 실시간 노출(INT-6).
- 비판 10건: 폴링 문서 불일치(INT-1), 중단 상태 미확정(INT-2), 취소 갭+daemon 지속(INT-3), 생성 중 전면 비활성(INT-4), 재시도 버튼 부재(INT-5), 빌드 취소 피드백 부재(INT-7), 자동 점프(INT-8), 재마운트 스크롤 리셋(INT-9), 모델 교체 블로킹(INT-10), 가이드 문구 중복(INT-11).
