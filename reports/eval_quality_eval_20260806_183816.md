# RAG Answer Quality Evaluation Report

- Date: 2026-08-06T18:39:14
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 512 / num_ctx: 4096
- Truncation threshold: 460.8 (0.9 x num_predict)
- --no-llm: True / testset_n: 1

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 4 |
| P@1 | 0.0 |
| MRR@5 | 0.0 |
| latency_question_count | 5 |
| avg TTFT (s) | None |
| avg TPS | None |
| avg answer chars | 0.0 |
| avg eval_count | None |
| truncated ratio | None (0건) |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | None (0건) |
| eval_count missing | 5건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | TTFT(s) | TPS | eval_count | truncated | overflow | judge | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.0 |  |  |  | 0 | 0 |  | 0.8992 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 |  |  |  | 0 | 0 |  | 0.0562 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 |  |  |  | 0 | 0 |  | 0.9083 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 |  |  |  |  |  | 0 | 0 |  |  |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | 0 | 0.0 |  |  |  | 0 | 0 |  | 0.0567 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None |  |  |  | 0 | 0 |  | 0.8551 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
