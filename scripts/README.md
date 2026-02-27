# RAG System Scripts Guide (스크립트 사용 가이드)

이 폴더는 RAG 시스템의 성능 측정, 품질 평가 및 유지보수를 위한 도구들을 포함하고 있습니다. 모든 스크립트는 프로젝트 루트 디렉토리에서 실행하는 것을 권장합니다.

## 🚀 시작하기 전 설정

모든 스크립트는 `src` 폴더를 모듈 경로로 인식해야 합니다. 실행 시 다음과 같이 환경 변수를 설정하세요.

### Windows (PowerShell)
```powershell
$env:PYTHONPATH="src"
python scripts/benchmarks/indexing_perf.py
```

### Linux / macOS
```bash
export PYTHONPATH=src
python scripts/benchmarks/indexing_perf.py
```

---

## 📊 성능 벤치마크 (`scripts/benchmarks/`)

시스템의 리소스 효율성과 처리 속도를 측정합니다.

| 도구명 | 용도 | 기대 결과 |
| :--- | :--- | :--- |
| **`indexing_perf.py`** | 대량 문서(5,000+) 인덱싱 성능 측정 | 인덱싱 소요 시간 및 메모리 점유율(SQ8 효율) |
| **`embedding_perf.py`** | 임베딩 모델별 생성 속도 비교 | 초당 임베딩 생성 수 (Tokens/sec) |
| **`parser_perf.py`** | PDF 마크다운 추출 엔진 속도 측정 | 페이지당 텍스트 추출 소요 시간 |
| **`rag_total_perf.py`** | 전체 RAG 파이프라인 E2E 시간 측정 | 첫 토큰 생성 시간(TTFT) 및 전체 응답 시간 |
| **`reranker_perf.py`** | 리랭커(FlashRank) 정렬 속도 측정 | 문서 수에 따른 리랭킹 지연 시간 |

---

## 🎯 품질 평가 (`scripts/evaluation/`)

RAG 시스템의 답변 정확도와 근거성을 정량적으로 평가합니다.

| 도구명 | 용도 | 실행 방법 |
| :--- | :--- | :--- |
| **`gen_testset.py`** | PDF에서 Q&A 테스트셋 생성 | `python scripts/evaluation/gen_testset.py` |
| **`quick_eval.py`** | **[핵심]** 데일리 고속 품질 평가 | `python scripts/evaluation/quick_eval.py` |
| **`compare_configs.py`** | 설정별(가중치 등) 품질 비교 실험 | `python scripts/evaluation/compare_configs.py` |
| **`evaluate_rag_quality.py`** | RAGAS 기반 상세 지표 산출 | `python scripts/evaluation/evaluate_rag_quality.py` |

> **Tip:** `quick_eval.py`는 LLM 채점(1-5점)과 의미론적 유사도를 동시에 측정하여 개발 과정에서 가장 빠르게 품질을 확인할 수 있는 도구입니다.

---

## 🛠️ 시스템 유지보수 (`scripts/maintenance/`)

데이터 정리 및 환경 관리를 위한 유틸리티입니다.

*   **`clean_artifacts.py`**: 로그, 임시 파일, 오래된 캐시를 일괄 정리합니다.
*   **`verify_integrity.py`**: 시스템 구성 요소 및 모델 파일의 무결성을 점검합니다.
*   **`migrate_cache_v1_to_v2.py`**: 이전 버전의 캐시 데이터를 최신 형식으로 변환합니다.

---

## 🔄 권장 워크플로우 (신규 문서 적용 시)

1.  **데이터 준비:** 평가할 PDF를 `tests/data/`에 배치.
2.  **테스트셋 생성:** `gen_testset.py`를 실행하여 질문-정답 쌍(CSV) 생성.
3.  **기본 품질 확인:** `quick_eval.py`를 실행하여 답변 품질 점수(Score) 확인.
4.  **성능 확인:** `indexing_perf.py`를 실행하여 메모리 사용량이 허용 범위인지 확인.
5.  **정밀 분석:** 품질이 낮을 경우 `evaluate_rag_quality.py`로 Faithfulness 등의 지표 상세 분석.

---

## 📂 결과 리포트 위치
모든 스크립트의 실행 결과(CSV, JSON, 리포트)는 프로젝트 루트의 **`reports/`** 폴더에 날짜별로 저장됩니다.
