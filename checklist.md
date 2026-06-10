# RAG 파이프라인 최적화 체크리스트

## Phase 1: LLM Grading Short-circuit
- [x] `src/api/schemas.py` 또는 관련 메타데이터에 `rerank_score` 필드 확인
- [x] `src/core/graph_builder.py`의 `grade_documents` 노드 수정 (Short-circuit 로직 추가)
- [x] 최상위 점수 0.85 이상 시 LLM 호출 스킵 여부 로그 확인
- [x] 속도 개선 수치 측정

## Phase 2: Dynamic Reranking Top-K
- [x] `src/core/graph_builder.py`의 `retrieve_and_rerank` 내 Score Gap 분석 로직 추가
- [x] 격차(Gap)에 따른 `dynamic_top_k` 결정 로직 구현
- [x] FlashRank 호출 시 동적 후보군 전달 확인

## Phase 3: Unified Indexing & Single-Pass Pruning
- [x] `src/core/semantic_chunker.py`에 중복 제거 로직 통합
- [x] `src/core/chunking.py`에서 불필요한 `IndexOptimizer` 중복 호출 제거
- [x] 인덱싱 시간 측정 및 대용량 문서 테스트
- [x] `graphify update .` 실행하여 지식 그래프 업데이트 (CLI 이슈 확인됨)

## Final Verification
- [x] 전체 파이프라인 통합 테스트 수행
- [x] `docs/superpowers/specs/2026-06-10-rag-pipeline-optimization.md` 문서 업데이트 및 완료 처리
