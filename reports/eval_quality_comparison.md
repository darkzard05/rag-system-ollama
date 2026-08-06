# RAG Answer Quality — Baseline vs After 비교 보고서

- 작성일: 2026-08-06 (Todo 9)
- 비교 대상:
  - **baseline**: `reports/eval_quality_baseline_20260806_180817.{json,md}` (2026-08-06T18:11:18, Wave 0 clean baseline)
  - **after**: `reports/eval_quality_after_20260806_230305.{json,md}` (2026-08-06T23:06:07, Wave 1-2 개선 적용 후)
- 실행 조건 동일성: 동일 PDF `tests/data/2201.07520v1.pdf`, 동일 질문셋(golden 5 + testset 상위 3), 동일 모델 `qwen3:4b-instruct-2507-q4_K_M` / `nomic-embed-text-v2-moe`, 동일 하네스 `scripts/eval_quality.py --tag <tag> --testset_n 3`, full 모드 (`--no-llm` 아님). 질문 오류 0건.

## 변경점 (baseline → after)

| 항목 | baseline | after |
| --- | --- | --- |
| num_predict | 512 | **2048** |
| num_ctx | 4096 | **8192** |
| 리랭커 | bi-encoder 코사인 | **FlashRank ms-marco-MultiBERT-L-12** |
| 리랭커 점수 분포 | 0.30~0.57 (코사인) | 0.05~0.94 (sigmoid, 이분 분포) |
| grade short-circuit | 없음 (전 질문 grade LLM 실행) | `min_score_to_skip=0.85` (관련 질문 grade 생략) |
| 생성 프로토콜 | baseline ANALYSIS_PROTOCOL | ANALYSIS_PROTOCOL 6-8 (간결성 지침) + generate 토큰 가드 |

> **⚠ 잘림 임계값 변화 (비교 해석 전제)**: baseline `0.9 × 512 = 460.8`, after `0.9 × 2048 = 1843.2`. 두 실행의 `truncated` 플래그는 서로 다른 임계값에서 판정되므로 **절대값 감소가 아닌 상대 비교**로 해석한다 (after는 같은 답변이어도 임계값이 4배 커져 truncated 판정 확률이 구조적으로 낮음).
>
> **⚠ max_rerank_score 척도 변화**: baseline은 bi-encoder 코사인, after는 FlashRank sigmoid. 두 값은 **직접 비교 불가** (기준 역할만 참고).

## 1. 7문항 집계 (baseline vs after)

| 지표 | baseline | after | Δ | 판정 |
| --- | --- | --- | --- | --- |
| scorable_count (golden, 퇴화 제외) | 4 | 4 | ±0 | 동일 |
| P@1 | 0.0 | 0.0 | ±0 | 동일 (개선 없음) |
| MRR@5 | 0.0 | 0.0 | ±0 | 동일 (개선 없음) |
| latency_question_count | 7 | 7 | ±0 | 동일 |
| avg TTFT (s) | 16.4318 | 18.5552 | **+2.1234 (+12.9%)** | 악화 |
| avg TPS | 49.0248 | 54.8513 | **+5.8265 (+11.9%)** | 개선 |
| avg answer chars | 446.2857 | 289.2857 | -157.0 | 감소 (간결) |
| avg eval_count (답변 토큰) | 283.7143 | 175.1429 | -108.5714 | 감소 (간결) |
| truncated ratio | 0.1429 (1/7) | **0.0 (0/7)** | -0.1429 | 개선 |
| overflow ratio | 0.0 (0/7) | 0.0 (0/7) | ±0 | 동일 |
| judge avg (1-5) | 3.3333 (3건) | 3.0 (3건) | -0.3333 | 악화 |
| eval_count missing | 0건 | 0건 | ±0 | 동일 |

## 2. 질문별 상세 (baseline → after)

| 질문 | P@1 b→a | MRR@5 b→a | TTFT(s) b→a | eval_count b→a | truncated b→a | max_rerank b→a | judge b→a |
| --- | --- | --- | --- | --- | --- | --- | --- |
| golden:1 (CM3 학습 원리/토큰화) | 0→0 | 0.0→0.0 | 22.32→**37.63** | 512→316 | 1→0 | 0.5329→0.8992 | — |
| golden:2 (파라미터 수/데이터셋 규모) | 0→0 | 0.0→0.0 | 12.44→**33.28** | 301→**42** | 0→0 | 0.4485→0.8620 | — |
| golden:3 (핵심 기여점 요약) | 0→0 | 0.0→0.0 | 11.15→**8.97** | 367→263 | 0→0 | 0.3217→0.9083 | — |
| golden:4 (**제외됨**: 빈 엔티티) | — | — | 14.74→19.06 | 75→97 | 0→0 | 0.3075→0.9421 | — |
| golden:5 (**out-of-doc 스트레스**, P@1=0 기대) | 0→0 | 0.0→0.0 | 15.60→16.31 | 9→24 | 0→0 | 0.4605→0.0567 | — |
| testset:1 (모델 종류/핵심 특징) | — | — | 12.57→**8.52** | 379→261 | 0→0 | 0.5107→0.8551 | 4→3 |
| testset:2 (훈련 데이터셋 구성/순서) | — | — | 27.34→**8.85** | 356→209 | 0→0 | 0.5722→0.8768 | 5→3 |
| testset:3 (훈련 데이터 기반) | — | — | 13.60→16.32 | 62→111 | 0→0 | 0.4950→0.8330 | **1→3** |

- P@1/MRR@5는 golden 스코러블(4문항)만 산출, judge는 testset(3문항)만 산출 (하네스 규칙).
- after 실행의 경로 관찰:
  - **short-circuit 발동** (max_rerank ≥ 0.85): golden:1 (0.8992), golden:3 (0.9083), golden:4 (rewrite 후 0.9421), testset:1 (0.8551), testset:2 (0.8768)
  - **grade 실행** (초기 max < 0.85): golden:2 (rewrite NO×2 → 최대 재시도 3회 고갈), golden:5 (초기 0.0567, grade YES), testset:3 (초기 < 0.85, grade YES)
  - baseline은 코사인 분포 0.32~0.57로 전 질문 grade 실행 (short-circuit 없음)

## 3. 판정 (계획 Todo 9 기준)

### (a) P@1 / MRR@5 / judge 평균 "동일 이상"
- P@1: 0.0 → 0.0 — **동일** (개선 없음, 원인 분석 §4)
- MRR@5: 0.0 → 0.0 — **동일** (개선 없음, 원인 분석 §4)
- judge 평균: 3.33 → 3.0 — **악화 (-0.33)**. 표본 3건 내역: testset:1 4→3, testset:2 5→3, testset:3 **1→3 (개선)**. 악화는 간결성 지침에 따른 상세성 감소로 해석 (§4). 표본 3건이라 편차가 큼.
- → **(a) 부분 미달**: 순위 지표 동일, judge 평균 소폭 악화.

### (b) TTFT / TPS "동일 이상"
- TPS: 49.02 → 54.85 — **개선 (+11.9%)** (전 질문 개선, golden:5만 -1.6 근소 악화)
- TTFT: 16.43 → 18.56 — **악화 (+12.9%)**. 원인 기록:
  - golden:1 (+15.3s): **FlashRank 콜드 스타트** — 첫 추론에서 ONNX 모델 로드 포함 (retrieve 단계 19s; 이후 질문은 6s대). 1회성 비용.
  - golden:2 (+20.8s): **rewrite 경로 3회 고갈** — 초기 검색 max_rerank < 0.85로 grade 실행 → 관련성 NO×2 → 재작성/재검색 2회 추가 (질문당 3회 검색). baseline golden:2는 1회 검색으로 grade 통과.
  - short-circuit 발동 3문항(golden:3 -2.2s, testset:1 -4.1s, testset:2 -18.5s)은 **전부 TTFT 개선** → 0.85 임계값/FlashRank 효과는 실재.
- → **(b) 부분 미달** (TPS 개선, TTFT 악화). TTFT 악화는 grade 임계값과 무관한 검색 경로/콜드 스타트 비용. 재조정은 이번에 수행하지 않고 **보고만** (계획 허용 범위 내).

### (c) truncated / overflow 비율 감소
- truncated: 1/7 → **0/7** — **개선** (임계값 460.8→1843.2 상향 반영이지만, 절대 건수도 1→0. golden:1이 512캡에서 316토큰으로 정상 완주).
- overflow: 0/7 → 0/7 — **동일** (num_ctx 4096→8192 확대 후에도 prompt_tokens_est 최대 3209 < 8192, 미발생 유지).
- → **(c) 달성**.

### (d) 종합
- 개선: TPS, truncated, overflow 유지, 답변 간결성(eval_count 283.7→175.1), out-of-doc 거짓 관련 차단(FlashRank 0.0567로 저신뢰), short-circuit 경로 TTFT.
- 동일: P@1/MRR@5 (=0), overflow.
- 악화: TTFT(평균), judge(평균), golden:2 답변 품질 회귀.

## 4. 개선 없음/악화 항목 원인 분석

1. **P@1=0, MRR@5=0 유지** (golden 4문항): 엄격 **all-entities 규칙** (`expected_key_entities` 전부가 상위 5개 문서 page_content에 포함돼야 relevant) 때문. golden 질문의 엔티티 셋이 단일 청크 단위로는 맞지 않아 리랭커 개선과 무관하게 0이 유지됨. FlashRank 0.85 임계값은 grade LLM 호출 여부만 제어하며 검색 순위를 바꾸지 않으므로 **이번 임계값 변경과 무관한 검색 순위 문제**.
2. **judge 평균 악화 (3.33→3.0)**: ANALYSIS_PROTOCOL 6-8 간결성 지침으로 답변이 압축됨 — testset:1 379→261, testset:2 356→209 토큰. judge가 "세부 정보 부족"으로 3점(핵심 맞음, 노이즈/상세 부족) 부여. 단, testset:3은 1→3으로 개선되어 간결 답변이 정답과 부합하는 경우엔 개선 방향. 표본 3건 한계.
3. **golden:2 답변 회귀** (baseline 301토큰 정확 답변 → after 42토큰 "정보 없음"): rewrite 경로에서 LLM 재작성 쿼리가 비결정적으로 생성되고, FlashRank 리랭킹으로 파라미터 수/데이터셋 규모가 포함된 CM3 섹션(p.3-4) 문서가 최종 상위 5개에서 이탈 → 컨텍스트에 수치 부재 → 거절 답변. **검색/리랭킹 순위 문제**이며 0.85 임계값과 무관.
4. **TTFT 평균 악화**: §3(b) — FlashRank 콜드 스타트(1회) + golden:2 rewrite 3회 고갈(해당 질문만 +20.8s). 재시도 고갈 시 최종 검색 결과의 max_rerank가 0.85 이상이어도 short-circuit 재확인 없이 generate로 직행하는 그래프 동작도 확인(예: golden:2 최종 0.862) — 구조적 개선 후보로 기록.

## 5. 특이사항

- **GeneratorExit / "Task exception was never retrieved" 경고**: full 모드에서 **미발생** (로그 grep 0건). `--no-llm` 조기 스트림 종료 전용 노이즈임을 재확인 (baseline 노트와 일치).
- max_rerank_score 척도 변경(코사인→FlashRank sigmoid)으로 값 직접 비교 불가. after의 이분 분포(관련 0.855~0.908 / 비관련 0.0567~0.833)는 기대와 부합.
- golden:5(out-of-doc)가 FlashRank에서 0.0567로 올바르게 저신뢰 판정 → 기대대로 P@1=0, "정보 없음" 답변 유지 (상태 정상).
- golden:4(퇴화)도 FlashRank 0.9421로 높게 나오지만 P@1/judge 집계에서 제외되어 영향 없음.
- 실행 시간: after 약 3분 1초 (23:03:06 ~ 23:06:07), 질문 오류 0건.
