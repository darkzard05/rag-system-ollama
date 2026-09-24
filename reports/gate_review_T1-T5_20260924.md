# Top5 Wave1-3 게이트리뷰 — 원자커밋 순서 확정 (T1→T2→T3→T5→T4)

> 원칙: commit/push/rebase/squash 절대금지, src\ 수정금지, 읽기·bash 측정만. 문서만 생성.

- 생성일: 2026-09-24
- 베이스라인 커밋: `a0778fc` (reports/perf_baseline_20260924_091935.md 생성일 커밋)
- 현재 브랜치: `main`
- 커버리지 게이트: `≥55%` (`--cov-fail-under=55`)
- 전체 파일 수: untracked 포함 6개 추가 + 4개 수정

---

## 1) 변경 스코프 (git diff --stat)

```
 config.yml                    |  2 +-   (k 25 → 15)
 src/core/async_reranker.py    | 54 ++++  (정규화 해시 + model키 + dim 가드)
 src/core/retriever_factory.py |  2 +-   (efSearch 128 → 64)
 src/core/semantic_chunker.py  | 32 ++    (gather 배치 + hit율 로그)
 4 files changed, 55 insertions(+), 35 deletions(-)
```

### 각 파일 diff 1~3줄 핵심

| 파일 | 핵심 diff (1~3줄) |
|------|-------------------|
| `config.yml:84` | `-      k: 25  # 충분한 후보군 확보를 위해 상향` → `+      k: 15  # k-chain Stage-1 단일 절단 (N×2 지배항=BM25 get_scores, RRF/dynamic_top_k 불변)` |
| `semantic_chunker.py:353-405` | `hit_rate = 1.0 - len(missing_texts)/len(texts)` 로그 추가, `asyncio.gather(*cache_manager.set(...))` 단일 `wait_for(30.0)` 병렬 배치 저장 (逐차 per-item wait_for 제거) |
| `async_reranker.py:45-48, 51-64, 127-148` | `_text_hash = " ".join(text.split())` 정규화, `model` 키 포함 `f"{model}:{hash}"`, metadata 우선 + `embedding_text_hash` + `dim` 일치 가드, module 캐시 분기 순서 교체 |
| `retriever_factory.py:139` | `-        ef_search = 128` → `+        ef_search = 64` (2단계 HNSW32,Flat 티어, 500 ≤ chunk < 5000) |

### untracked (커밋 대상에 포함 여부 판단 필요)

```
?? repomix-output.xml
?? reports/eval_quality_k15_20260924_092717.json  ← T3 증거, 포함
?? reports/eval_quality_k15_20260924_092717.md    ← T3 증거, 포함
?? reports/perf_baseline_20260924_091935.md       ← 베이스라인 측정 문서, 포함
?? scripts/reports/                               ← 비어있음/확인 필요, 제외 권장
?? tests/unit/test_perf_chunk_embed.py            ← T1 TDD, 포함
?? tests/unit/test_perf_rerank_reuse.py           ← T2 TDD, 포함
?? temp_test_ui.py                                ← 무관, 제외
```

> `repomix-output.xml`, `temp_test_ui.py`, `scripts/reports/`는 Top5 스코프 밖이므로 원자커밋에서 제외한다. 포함하면 히스토리 오염.

---

## 2) 정적/보안/타입 QA 리포트 (읽기전용 실행)

| 명령 | 결과 | 판정 |
|------|------|------|
| `ruff check .` | `All checks passed!` | PASS |
| `ruff format --check .` | `91 files already formatted` | PASS |
| `mypy src` | `Success: no issues found in 90 source files` | PASS |
| `bandit -r src/ -ll` | `No issues identified. (High 0, Medium 0, Low 14 filtered by -ll)` · Total lines 17806 · skipped 0 | PASS |
| `pytest tests/unit --collect-only` | `905 tests collected` | PASS |
| `pytest tests/unit -q --tb=no --no-cov` | `1 failed, 904 passed, 2 warnings` | **FAIL** (원인: T5) |
| `pytest tests/unit --cov=src --cov-fail-under=55` | `78.51%` (10192 stmts, 2190 miss) — Required 55% reached, 1 failed / 904 passed | **PASS (커버리지는 PASS, 전체는 1 failed로 RED)** |
| `pytest tests/integration/test_pipeline_build_once.py -v` | `1 passed` | PASS |
| `pytest -k chunk` | `78 passed, 827 deselected` | PASS |
| `pytest -k rerank` | `21 passed, 884 deselected` | PASS |
| `pytest tests/unit/test_perf_chunk_embed.py tests/unit/test_perf_rerank_reuse.py -v` | `9 passed (5 chunk + 4 rerank)` | PASS |

### 단일 실패 상세

```
FAILED tests/unit/test_index_nprobe_configured.py::test_hnsw_efsearch_applied_without_faiss_gpu_symbols
  assert 64 == 128  (hnsw.efSearch)
  - 테스트는 n=3000 (2단계 HNSW32,Flat, ef_search=128) 가정
  - 현 코드는 ef_search=64 (T5 변경) → 기대값 불일치
```

- 실패는 T5 변경의 의도된 결과이며 코드 버그가 아니다. 테스트 기대값을 64로 갱신해야 한다.
- 그 외 904개 통과는 chunk/rerank/그래프/세션 등 전 영역에서 회귀 없음을 의미한다.

---

## 3) T1~T5 증거 대조

### T1 — chunk embed gather 배치 + hit율 (Wave1)

- **파일**: `src/core/semantic_chunker.py:310-453` → 변경 352~419
- **증거**:
  - hit율 로그: `f"[Chunker] {len(missing_texts)}개 ... hit율: {hit_rate:.1%}"` (353~355)
  - 단일 `wait_for + gather` 병렬 저장 (406~418): `unique_new` deduplication 후 `asyncio.gather(*set(...))` 1회 대기. 이전 逐차 per-item `wait_for` 대비 miss 다수 시 `t_embed` 지배 구간 제거.
  - TDD: `tests/unit/test_perf_chunk_embed.py` 5개 전부 PASS
    - `test_cache_save_is_batched_outside_single_wait`: miss 6건 × set 0.1s → `max_in_flight == 6` 단언 (병렬성 증명, 타이머 플레이크 제거)
    - `test_miss_logs_hit_rate`: 2/4 워밍 → `hit율 50.0%` 로그 단언
    - `test_warm_cache_skips_embed_and_is_fast`: warm-cache에서 `embed_calls==1` + wall-time < 2000ms
    - `test_as_valid_vector_guards`, `test_resolve_expected_dim_prefers_new`: 가드 유지
  - 광역: `pytest -k chunk` 78 passed (기존 청커 회귀 없음)
  - 로그 계약: `[Chunker] N개 문장 신규 임베딩 생성 중 (Batch Size: ..., hit율: ...)` — 기존 `missing_texts` 기반 M_unique 수집 유지

### T2 — rerank 정규화 + model키 + dim 가드 (Wave1)

- **파일**: `src/core/async_reranker.py:45-175`
- **증거**:
  - 정규화 해시: `"_text_hash = md5(\" \".join(text.split()))"` — chunker `norm_text = " ".join(text.split())` (semantic_chunker.py:329)와 동일 규약
  - model키 격리: `_get_cached_emb(text, model)`, `_set_cached_emb(text, vec, model)` — `model = getattr(embedder, "model"|"model_name")`
  - dim 불일치 시 재계산: `dim=int(query_vec.shape[0])` 전달, `len(meta_vec)==dim` / `len(cached_vec)==dim` 가드
  - 메타데이터 우선 + 정규화 해시 동등성: `meta_hash == _text_hash(page_content)` 일 때만 재사용, 아니면 module 캐시 → 재임베딩
  - 스코프: `embedding_text_hash` 메타데이터 신규 기록 (173)
  - TDD: `tests/unit/test_perf_rerank_reuse.py` 4개 PASS
    - `test_text_hash_normalizes_like_chunker`: `"a  b\n c" == "a b c"` 해시 동등
    - `test_full_metadata_hit_skips_embed_documents`: 전원히트 시 `doc_calls==0` + `scores` 내림차순 (recall@5 불변)
    - `test_dim_mismatch_recomputes`: stale `len==1` 벡터 → `doc_calls==1` 재계산
    - `test_model_key_isolates_cache`: model-a 캐시가 model-b에 재사용 안 됨
  - 광역: `pytest -k rerank` 21 passed

### T3 — k-chain Stage-1 단일 절단 k=15 + 품질 평가 (Wave2)

- **파일**: `config.yml:84` `k: 25 → 15` + 리포트 2종
- **증거**:
  - 코멘트: `k-chain Stage-1 단일 절단 (N×2 지배항=BM25 get_scores, RRF/dynamic_top_k 불변)` — BM25 `get_scores`가 N×2 지배항이므로 k 절단이 직접적 비용 절감
  - RRF/dynamic_top_k 불변: `dynamic_top_k: {gap 0.003, min 12, max 18}` 유지, `ensemble_weights [0.4,0.6]` 유지, `grading.top_k 5` 유지
  - 품질 측정: `reports/eval_quality_k15_20260924_092717.json/.md` (`--no-llm` retrieval-only)
    - meta: `tag k15`, `pdf ../tests/data/2201.07520v1.pdf`, `no_llm True`, `testset_n 3`, `golden_total 5`
    - aggregates: `scorable 4`, `P@1 0.25`, `MRR@5 0.3333`, `context_recall 0.5556 (3건)`, `degenerate 1건(row4 빈 엔티티 제외)`
    - per-question: golden row1 P@1 0 MRR 0.3333 / row2 0/0.0 / row3 0/0.0 / row5(out-of-doc) 1/1.0, testset 3건 P@1 null (golden 미포함)
    - 한계: `--no-llm`이므로 TTFT/TPS/judge/faithfulness는 None (LLM 생성 미수행), `eval_count missing 7건`
  - 베이스라인 문서: `reports/perf_baseline_20260924_091935.md` — N/S/r/t_embed/search_ms/T_llm/short-circuit%/cache hit% 수집법 정의, `k=25` 베이스라인으로 T3 전후 비교 근거 제공
  - 판정: 측정은 존재하나 `P@1 0.25`는 절대값으로는 낮음. 베이스라인 `k=25` 대비 동 리포트가 없으므로 k=15의 recall 유지 여부는 상대 비교 불가 — 베이스라인 재측정(동일 pdf, 동일 --no-llm)으로 델타를 내야 한다. 현재는 측정 인프라 구축 단계로 봐야 한다.

### T4 — W=1 유지 (동시성 상한, Wave3)

- **파일**: `config.yml:52` `max_concurrent_inference: 1` (변경 없음, 의도적 유지)
- **증거**:
  - 코멘트: `기본값 1 = 직렬 추론. 1보다 크게 설정하면 OOM 위험` + `host_pressure_threshold 85.0` 초과 시 자동 1로 강등
  - `perf_baseline` 문서의 W=1,2,4 큐대기 측정 명령도 `W=1`을 기본으로 명시
  - 커밋 diff 없음 → 별도 소스 변경 없이 게이트 검증만 수행. 향후 T4 단독 커밋은 문서/코멘트 보강 또는 검증 스크립트 추가로 채우거나, T3와 합쳐도 되나 순서상 T5 이후 마지막으로 배치해 W 상한 정책이 최종 상태임을 명시한다.

### T5 — efSearch 128 → 64 (2단계 HNSW, Wave2)

- **파일**: `src/core/retriever_factory.py:139` `ef_search = 64`
- **증거**:
  - 대상: `chunk_count < q_threshold (5000)` 2단계 `HNSW32,Flat` 티어만 영향. 3단계(5000~20000)는 `ef 256` 유지, 1단계 Flat/4단계 IVF는 `ef 0` 유지 — 변경 범위 최소
  - 기대 효과: HNSW 탐색 폭 50% 축소 → search_ms 감소. 주장 `-33%`는 코드 레벨에서 직접 측정되지 않았으나 efSearch가 선형 탐색 비용에 비례하므로 방향성은 타당. 실측은 `benchmarks/bench_query_latency.py` 또는 `eval_quality --no-llm`의 `search_ms` 로그로 검증해야 한다.
  - 테스트 충돌: `test_hnsw_efsearch_applied_without_faiss_gpu_symbols`는 128 기대 → 64로 실패. 이는 T5 의도이므로 테스트 갱신이 필수. IVF nprobe(16) 및 3단계 ef 256 관련 테스트는 통과하므로 recall 핵심 경로는 유지.
  - 동일 recall 주장은 IVf nprobe 미변경 + 2단계가 500~5000 소규모 티어로 HNSW recall 손실이 제한적이라는 가정에 의존. 실측 recall은 T3 리포트와 동일 pdf로 `k15 × ef64` 조합에서 재측정해야 확정된다.

---

## 4) 커밋 리스트 — 원자커밋 순서 확정

> 스타일: `perf(<area>): <imperative summary> (refs Tn)` — Conventional Commits perf 타입, scope는 area, 본문 refs로 Tn 추적.
> 순서: **T1 → T2 → T3 → T5 → T4** (CR-004 확정). 의존성: T1/T2는 독립적 임베딩 경로, T3(k)는 검색 N을 결정하므로 T5(ef)보다 선행, T4(W)는 전역 동시성 상한으로 마지막에 최종 상태 고정.

### Commit 1 — T1

```
perf(chunker): batch embed-cache saves with single gather and hit-rate logging (refs T1)
```

- **파일**:
  - `src/core/semantic_chunker.py`
  - `tests/unit/test_perf_chunk_embed.py`
- **본문 (제안)**:
  ```
  Replace per-item wait_for cache saves with single
  asyncio.wait_for(gather(*set())) batch and add hit-rate
  debug log. Keeps persist_to_disk=True contract and
  M_unique accounting.

  Tests: 78 chunk passed (incl. 5 new TDD).
  Evidence: hit율 50.0% log, max_in_flight==N parallel.
  ```
- **QA (해당 커밋만)**:
  ```
  ruff check src/core/semantic_chunker.py tests/unit/test_perf_chunk_embed.py
  mypy src/core/semantic_chunker.py
  python -m pytest tests/unit/test_perf_chunk_embed.py tests/unit/test_semantic_chunker.py -v
  python -m pytest tests/unit -k chunk -q --tb=no --no-cov
  ```

### Commit 2 — T2

```
perf(reranker): normalize text hash, isolate cache by model and guard dim mismatch (refs T2)
```

- **파일**:
  - `src/core/async_reranker.py`
  - `tests/unit/test_perf_rerank_reuse.py`
- **본문**:
  ```
  Normalize _text_hash via " ".join(text.split()) (same as
  chunker norm_text), add model key to module cache, and
  require embedding_text_hash + dim match before metadata
  reuse. Prevents stale cross-model/dim hits.

  Tests: 21 rerank passed (incl. 4 new TDD), recall@5 invariant.
  ```
- **QA**:
  ```
  ruff check src/core/async_reranker.py tests/unit/test_perf_rerank_reuse.py
  mypy src/core/async_reranker.py
  python -m pytest tests/unit/test_perf_rerank_reuse.py -v
  python -m pytest tests/unit -k rerank -q --tb=no --no-cov
  ```

### Commit 3 — T3

```
perf(retriever): cut Stage-1 candidate pool k 25→15 (refs T3)
```

- **파일**:
  - `config.yml` (k 15)
  - `reports/eval_quality_k15_20260924_092717.json`
  - `reports/eval_quality_k15_20260924_092717.md`
  - `reports/perf_baseline_20260924_091935.md`
- **본문**:
  ```
  Single-cut k-chain Stage-1: k 25→15 reduces dominant
  BM25 get_scores (N×2) cost while RRF/dynamic_top_k
  (12–18) and rerank top_k 5 stay invariant.

  Evidence: k15 --no-llm eval P@1 0.25 MRR 0.3333
  context_recall 0.5556 (scorable 4, 1 degenerate excluded).
  Baseline doc added for W/k comparison methodology.
  ```
- **QA**:
  ```
  ruff check config.yml  # yaml은 ruff 대상 아님, 형식 수동 확인
  python scripts/eval_quality.py --tag k15 --no-llm --testset_n 3  # 재현
  python -m pytest tests/integration/test_pipeline_build_once.py -v
  ```

### Commit 4 — T5

```
perf(index): lower HNSW efSearch 128→64 for 500–5000 tier (refs T5)
```

- **파일**:
  - `src/core/retriever_factory.py`
  - `tests/unit/test_index_nprobe_configured.py` ← **기대값 갱신 필요 (64)**
- **본문**:
  ```
  Halve efSearch for HNSW32,Flat (chunk<5000) tier from
  128 to 64. 3rd tier (ef 256) and IVF nprobe (16) unchanged.
  Expected -33% search_ms at same recall (to be verified
  via bench_query_latency / eval_quality search_ms).

  Test: update test_hnsw_efsearch_applied_without_faiss_gpu_symbols
  to expect 64.
  ```
- **QA**:
  ```
  ruff check src/core/retriever_factory.py
  mypy src/core/retriever_factory.py
  python -m pytest tests/unit/test_index_nprobe_configured.py -v
  python -m pytest tests/unit/test_rag_performance.py -v
  ```

### Commit 5 — T4

```
perf(concurrency): keep max_concurrent_inference at 1 (refs T4)
```

- **파일**:
  - `config.yml` (주석/검증 보강이 있으면 포함, 없으면 빈 커밋 대신 T3에 squash 금지 — W 상한 최종 상태 명시용)
  - 또는 `docs/`/`reports/`에 W=1 검증 로그 추가
- **본문**:
  ```
  Keep W=1 serial inference bound; host_pressure 85% auto-
  degrades to 1. Validated that speculative overlap stays
  disabled at W=1 (no behavior change) and OOM risk bound.

  Evidence: config max_concurrent_inference=1, host_pressure 85.0.
  ```
- **QA**:
  ```
  grep -n "max_concurrent_inference" config.yml
  python -m pytest tests/unit/test_warmup_concurrency.py -v -k "concurrency or warmup"
  python -m pytest tests/integration/test_pipeline_build_once.py -v
  ```

> 참고: T4는 소스 diff가 없으므로 단독 커밋이 비어 보일 수 있다. 대안으로 T3 커밋 본문에 `Refs T3, T4`를 명시하고 T4를 문서 커밋으로 남기는 것도 허용되나, 순서 게이트는 T5→T4를 유지해야 하므로 T4를 마지막에 두는 것이 정책 가시성에 유리하다.

---

## 5) 커밋당 QA 명령 요약 (복붙용)

```bash
# C1 T1
ruff check src/core/semantic_chunker.py tests/unit/test_perf_chunk_embed.py
mypy src/core/semantic_chunker.py
python -m pytest tests/unit/test_perf_chunk_embed.py tests/unit/test_semantic_chunker.py -v
python -m pytest tests/unit -k chunk -q --tb=no --no-cov

# C2 T2
ruff check src/core/async_reranker.py tests/unit/test_perf_rerank_reuse.py
mypy src/core/async_reranker.py
python -m pytest tests/unit/test_perf_rerank_reuse.py -v
python -m pytest tests/unit -k rerank -q --tb=no --no-cov

# C3 T3
python scripts/eval_quality.py --tag k15 --no-llm --testset_n 3
python -m pytest tests/integration/test_pipeline_build_once.py -v

# C4 T5
ruff check src/core/retriever_factory.py
mypy src/core/retriever_factory.py
python -m pytest tests/unit/test_index_nprobe_configured.py -v
python -m pytest tests/unit/test_rag_performance.py -v

# C5 T4
python -m pytest tests/unit/test_warmup_concurrency.py -v -k "concurrency or warmup"
python -m pytest tests/integration/test_pipeline_build_once.py -v

# 전체 게이트 (마지막 커밋 후)
ruff check .
ruff format --check .
mypy src
bandit -r src/ -ll
python -m pytest tests/unit -q --tb=no --no-cov
python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-fail-under=55
python -m pytest tests/integration/test_pipeline_build_once.py -v
```

---

## 6) 전체 그린 여부 판정

### 판정: **RED — GATE BLOCK**

| 게이트 | 상태 | 비고 |
|--------|------|------|
| ruff check | GREEN | All checks passed |
| ruff format | GREEN | 91 files formatted |
| mypy src | GREEN | 90 files, no issues |
| bandit | GREEN | No High/Medium, Low만 14 ( -ll 필터) |
| pytest unit --no-cov | **RED** | 1 failed / 904 passed — T5 기대값 불일치 |
| pytest --cov ≥55% | **GREEN** | 78.51% ≥ 55%, 10192/2190, 1 failed는 T5 기대값 불일치 (커버리지는 PASS) |
| integration build_once | GREEN | 1 passed |
| T1 evidence | GREEN | 78 chunk passed + 5 TDD |
| T2 evidence | GREEN | 21 rerank passed + 4 TDD |
| T3 evidence | **YELLOW** | 리포트 존재하나 k25 베이스라인 대비 델타 없음, 절대값 P@1 0.25는 낮음 — 비교 재측정 필요 |
| T4 evidence | GREEN | W=1 유지 확인 |
| T5 evidence | **RED** | ef64 테스트 실패, recall 동일/ -33% 실측 로그 없음 |

### 차단 사유 (Top2)

1. **T5 테스트 실패 1건** — `test_hnsw_efsearch_applied_without_faiss_gpu_symbols`가 128을 기대하나 코드는 64. 커밋 전 테스트 기대값을 64로 갱신하거나 파라미터화된 테스트로 교체해야 한다. 현 상태로는 CI가 RED. (커버리지는 78.51%로 PASS이므로 단일 원인에 의한 RED)
2. **T3/T5 성능 실측 부재** — k15와 ef64의 latency/recall 델타가 코드 주석/리포트에만 있고 `search_ms`/`total_ms` 로그 기반 실측 비교가 없다. `bench_query_latency.py` 또는 `eval_quality`의 TIMING 로그로 k25×ef128 대비 k15×ef64 비교를 추가해야 가상 수치 논란을 피할 수 있다.

### 커밋 전 필수 조치

- [ ] `tests/unit/test_index_nprobe_configured.py:155` 기대값 `128 → 64` 수정 (또는 `pytest.mark.parametrize`로 64/256 티어 분리) — 해소 시 905 passed, 커버리지 78.51% 유지로 전체 GREEN
- [x] `python -m pytest tests/unit --cov=src --cov-report=term --cov-fail-under=55` 재실행 (180s) — `78.51% PASS` 확인 (10192 stmts)
- [ ] k25 베이스라인 `eval_quality --no-llm` 1회 추가 측정으로 T3 델타 확보 (동일 pdf, 동일 testset_n)
- [ ] ef64 search_ms 실측 로그 1회 수집 (`[RAG][RETRIEVE][TIMING] search_ms=...`)

위 4건 해소 후 전체 그린으로 전환 가능. 해소 전까지는 본 게이트리뷰 문서만으로 커밋 순서를 확정하고, 실제 `git commit`은 수행하지 않는다.

---

## 7) 부록 — 원시 명령 로그 발췌

```
$ git diff --stat
 config.yml                    |  2 +-
 src/core/async_reranker.py    | 54 +++++
 src/core/retriever_factory.py |  2 +-
 src/core/semantic_chunker.py  | 32 ++

$ ruff check .          → All checks passed!
$ ruff format --check . → 91 files already formatted
$ mypy src              → Success: no issues found in 90 source files
$ bandit -r src/ -ll    → No issues identified.
$ pytest -k chunk       → 78 passed
$ pytest -k rerank      → 21 passed
$ pytest tests/unit -q --no-cov → 1 failed, 904 passed
$ pytest integration/test_pipeline_build_once → 1 passed
```

---

*본 문서는 src\ 수정 없이 읽기·bash 측정만으로 작성되었으며, 커밋 메시지 규약 `perf(<area>): … (refs Tn)`과 순서 T1→T2→T3→T5→T4를 고정한다. 실제 커밋 생성은 게이트 RED 해소 후 별도 승인 하에 진행한다.*
