# RAG System Scripts Guide (스크립트 사용 가이드)

이 폴더는 RAG 시스템의 성능 측정, 품질 평가 및 유지보수를 위한 운영 도구들을 포함합니다. 모든 스크립트는 프로젝트 루트 디렉토리에서 실행하는 것을 권장합니다.

> **2026-09 정리:** 일회성 진단/검증 스크립트는 `scripts/archive/`로 이동했고, pytest 자격이 있는 것은 `tests/`로 승격했습니다. 이 README는 현재 운영 유지되는 스크립트만 나열합니다.

## 🚀 시작하기 전 설정

운영 스크립트는 `src` 패키지를 직접 참조하므로 실행 시 `src`를 모듈 경로에 노출해야 합니다.

### Windows (PowerShell)
```powershell
$env:PYTHONPATH="src"
python scripts/benchmarks/total_rag_perf.py
```

### Linux / macOS
```bash
export PYTHONPATH=src
python scripts/benchmarks/total_rag_perf.py
```

---

## 📊 성능 벤치마크 (`scripts/benchmarks/`)

시스템의 리소스 효율성과 처리 속도를 측정합니다.

| 도구명 | 용도 | 기대 결과 |
| :--- | :--- | :--- |
| **`total_rag_perf.py`** | 전체 RAG 파이프라인 E2E 시간 측정 | 첫 토큰 생성 시간(TTFT) 및 전체 응답 시간 |
| **`total_load_perf.py`** | 문서 로드/인덱싱 시간 측정 | 로드 소요 시간 및 메모리 점유율 |
| **`embedding_perf.py`** | 임베딩 모델별 생성 속도 비교 | 초당 임베딩 생성 수 (Tokens/sec) |
| **`parser_perf.py`** | PDF 마크다운 추출 엔진 속도 측정 | 페이지당 텍스트 추출 소요 시간 |
| **`quality_eval.py`** | 리랭커/품질 평가 벤치마크 | 리랭킹 지연 시간 및 품질 지표 |
| **`bench_query_latency.py`** | 쿼리 지연 시간 벤치마크 | 쿼리별 대기 시간/TTFT 분포 |
| **`benchmark_reranker.py`** | 리랭커 성능 프로파일링 | 리랭킹 지연 시간 및 메모리 |
| **`benchmark_aggregator.py`** | 벤치마크 결과 취합 | 보고서 집계 자료 |
| **`compare_qwen_models.py`** | Qwen 모델 간 지연/품질 비교 | 모델별 벤치마크 비교표 |
| **`bench_params.py`** | 공용 벤치마크 상수 | (모듈, 직접 실행 아님) |
| **`e2e_performance_benchmark.py`** | E2E 성능 벤치마크 | 전체 응답 지연 및 처리량 |

> 과거 일회성 벤치마크(`benchmark_parallel_retrieval`, `benchmark_serialization`, `benchmark_streaming_sim`, `benchmark_ttft`, `test_retriever_leak` 등)는 `scripts/benchmarks/archive/`에 보관되어 있습니다.

---

## 🎯 품질 평가 (`scripts/evaluation/`)

RAG 시스템의 답변 정확도와 근거성을 정량적으로 평가합니다.

| 도구명 | 용도 | 실행 방법 |
| :--- | :--- | :--- |
| **`gen_testset.py`** | PDF에서 Q&A 테스트셋 생성 | `python scripts/evaluation/gen_testset.py` |
| **`quick_eval.py`** | **[핵심]** 데일리 고속 품질 평가 | `python scripts/evaluation/quick_eval.py` |
| **`compare_configs.py`** | 설정별(가중치 등) 품질 비교 실험 | `python scripts/evaluation/compare_configs.py` |
| **`eval_advanced_retrieval.py`** | 검색/리랭킹 상세 지표 산출 | `python scripts/evaluation/eval_advanced_retrieval.py` |
| **`eval_quality.py`** | 종합 품질 평가 하네스 | `python scripts/evaluation/eval_quality.py` |
| **`eval_grader.py`** | 답변 채점 도구 | `python scripts/evaluation/eval_grader.py` |
| **`eval_retrieval.py`** | 검색 성능 지표 산출 | `python scripts/evaluation/eval_retrieval.py` |
| **`simple_eval.py`** | 간편 품질 점검 | `python scripts/evaluation/simple_eval.py` |
| **`eval_highlight_precision.py`** | 하이라이트 정밀도 평가 | `python scripts/evaluation/eval_highlight_precision.py` |
| **`evaluate_diverse_questions.py`** | 다양한 질의 유형 평가 | `python scripts/evaluation/evaluate_diverse_questions.py` |

> **Tip:** `quick_eval.py`는 LLM 채점(1-5점)과 의미론적 유사도를 동시에 측정하여 개발 과정에서 가장 빠르게 품질을 확인할 수 있는 도구입니다.

---

## 🛠️ 시스템 유지보수 (`scripts/maintenance/`)

CI·코드 품질·환경 관리를 위한 유틸리티입니다.

| 도구명 | 용도 |
| :--- | :--- |
| **`run_ci_checks.py`** | CI test stage 단일 소스오브트루스 (ruff/mypy/coverage/integration) |
| **`check_imports.py`** | scripts/tests first-party import 스윕 백스톱 |
| **`verify_integrity.py`** | 시스템 구성 요소 및 모델 파일 무결성 점검 |
| **`verify_requirements.py`** | requirements.txt 버전/상태 검증 |
| **`verify_ci_logic.py`** | CI 로직 사양 검증 |
| **`pip_audit_hook.py`** | pre-commit pip-audit 훅 |
| **`clean_artifacts.py`** | 로그/임시/캐시 정리 |
| **`export_openapi.py`** | API OpenAPI 스키마 내보내기 |
| **`update_readme.py`** | README 자동 갱신 도구 |
| **`migrate_cache_v1_to_v2.py`** | 이전 캐시 형식을 최신 버전으로 변환 |

---

## 🔬 모델 검증 (`scripts/verification/`)

| 도구명 | 용도 |
| :--- | :--- |
| **`probe_qwen3_thinking.py`** | Qwen3 모델 사고(reasoning) 동작 검증 |

---

## 🖥️ 루트 운영 도구

| 도구명 | 용도 |
| :--- | :--- |
| **`verify_e2e_all.py`** | Streamlit E2E 통합 검증 (DOM/스크롤/입력) — `verify_dom_structure`·`verify_ui_scrolling`·`test_chat_scroll`를 대체 |
| **`visual_qa_automation.py`** | 시각적 QA 자동화 |
| **`layout_probe.py`** | Streamlit 레이아웃 진단 운영 도구 |
| **`kill_streamlit.py`** | 실행 중인 Streamlit 프로세스 종료 |

---

## 📁 보관 (`scripts/archive/`)

일회성/라이브 의존 검증 스크립트가 여기 보관됩니다. 직접 실행을 전제로 하지 않으며, 재사용 시 참고용으로만 보존합니다.

- `test_full_pipeline.py` (E2E 라이브 Ollama — `python scripts/archive/test_full_pipeline.py`로 수동 QA 가능)
- `test_embedding_v2.py`, `test_section_filtering.py`, `verify_chunking_efficiency.py` (라이브 Ollama/PDF 의존)
- `test_retriever_leak.py` (실제 RSS 메모리 검증 — `test_resource_eviction`이 mock으로 대체)
- 그 외 일회성 verify/평가 스크립트 13개

---

## 🔄 권장 워크플로우 (신규 문서 적용 시)

1.  **데이터 준비:** 평가할 PDF를 `tests/data/`에 배치.
2.  **테스트셋 생성:** `gen_testset.py`를 실행하여 질문-정답 쌍(CSV) 생성.
3.  **기본 품질 확인:** `quick_eval.py`를 실행하여 답변 품질 점수(Score) 확인.
4.  **성능 확인:** `total_rag_perf.py`를 실행하여 메모리 사용량이 허용 범위인지 확인.
5.  **정밀 분석:** 품질이 낮을 경우 `eval_advanced_retrieval.py`로 Faithfulness 등의 지표 상세 분석.

---

## 📂 결과 리포트 위치
모든 스크립트의 실행 결과(CSV, JSON, 리포트)는 프로젝트 루트의 **`reports/`** 폴더에 날짜별로 저장됩니다.
