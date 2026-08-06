# RAG Answer Quality Evaluation Report

- Date: 2026-08-06T18:11:18
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
| avg TTFT (s) | 16.4318 |
| avg TPS | 49.0248 |
| avg answer chars | 446.2857 |
| avg eval_count | 283.7143 |
| truncated ratio | 0.1429 (1건) |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 3.3333 (3건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | TTFT(s) | TPS | eval_count | truncated | overflow | judge | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.0 | 22.3183 | 57.4834 | 512 | 1 | 0 |  | 0.5329 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 12.4443 | 53.5321 | 301 | 0 | 0 |  | 0.4485 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 | 11.1504 | 45.7326 | 367 | 0 | 0 |  | 0.3217 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | 14.7356 | 44.1262 | 75 | 0 | 0 |  | 0.3075 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | 0 | 0.0 | 15.5965 | 55.5096 | 9 | 0 | 0 |  | 0.4605 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | 12.573 | 44.0321 | 379 | 0 | 0 | 4 | 0.5107 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | 27.3418 | 42.9235 | 356 | 0 | 0 | 5 | 0.5722 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | 13.5982 | 43.9604 | 62 | 0 | 0 | 1 | 0.495 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
