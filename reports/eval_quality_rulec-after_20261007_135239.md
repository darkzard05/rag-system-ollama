# RAG Answer Quality Evaluation Report

- Date: 2026-10-07T13:57:23
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: False / testset_n: 3

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 3 |
| P@1 | 0.3333 |
| MRR@5 | 0.5 |
| context_recall (golden) | 0.75 (3건) |
| latency_question_count | 7 |
| avg TTFT (s) | 17.2572 |
| avg TPS | 38.1839 |
| avg answer chars | 909.0 |
| avg eval_count | 396.4286 |
| truncated ratio | 0.0 (0건) |
| parse_failed / echo / over_budget | 0 / 0 / 0건 |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 3.6667 (3건) |
| faithfulness avg (1-5) | 3.6667 (3건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | parse | echo | over_b | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 1 | 1.0 | 0.75 | 7.0527 | 46.9294 | 620 | 0 | 0 | 0 | 0 | 0 |  |  | 0.8838 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 | 40.8662 | 42.6573 | 317 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0537 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.5 | 1.0 | 9.2907 | 38.0014 | 453 | 0 | 0 | 0 | 0 | 0 |  |  | 0.7738 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | None | 51.0507 | 41.2422 | 29 | 0 | 0 | 0 | 0 | 0 |  |  | 0.08 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | None | None | None | 38.489 | 37.3054 | 519 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0573 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None | 9.305 | 34.7627 | 321 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.8512 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | None | 8.4694 | 33.6377 | 343 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.8307 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | None | 7.3277 | 33.9933 | 202 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.7892 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
