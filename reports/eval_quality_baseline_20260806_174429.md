# RAG Answer Quality Evaluation Report

- Date: 2026-08-06T17:47:05
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 512 / num_ctx: 4096
- Truncation threshold: 460.8 (0.9 x num_predict)
- --no-llm: False / testset_n: 3

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 4 |
| P@1 | 0.0 |
| MRR@5 | 0.0 |
| latency_question_count | 7 |
| avg TTFT (s) | 9.3169 |
| avg TPS | 51.7876 |
| avg answer chars | 592.8571 |
| avg eval_count | 382.1429 |
| truncated ratio | 0.4286 (3건) |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 4.6667 (3건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | TTFT(s) | TPS | eval_count | truncated | overflow | judge | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.0 | 37.5606 | 58.7603 | 512 | 1 | 0 |  | 0.5329 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 11.125 | 51.6204 | 414 | 0 | 0 |  | 0.4485 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 | 12.854 | 45.0796 | 374 | 0 | 0 |  | 0.3217 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | 19.5525 | 45.779 | 318 | 0 | 0 |  | 0.3581 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | 0 | 0.0 | 0.8207 | 70.6416 | 9 | 0 | 0 |  | 0.3581 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | 0.8194 | 45.1539 | 512 | 1 | 0 | 4 | 0.3581 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | 1.0021 | 45.4545 | 512 | 1 | 0 | 5 | 0.3581 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | 1.0367 | 45.803 | 342 | 0 | 0 | 5 | 0.3581 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
