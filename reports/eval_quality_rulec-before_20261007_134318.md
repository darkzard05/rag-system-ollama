# RAG Answer Quality Evaluation Report

- Date: 2026-10-07T13:51:51
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: False / testset_n: 3

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 2 |
| P@1 | 0.0 |
| MRR@5 | 0.25 |
| context_recall (golden) | 0.75 (2건) |
| latency_question_count | 6 |
| avg TTFT (s) | 21.9208 |
| avg TPS | 41.1748 |
| avg answer chars | 1270.8333 |
| avg eval_count | 519.6667 |
| truncated ratio | 0.0 (0건) |
| parse_failed / echo / over_budget | 0 / 0 / 0건 |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 3.6667 (3건) |
| faithfulness avg (1-5) | 3.6667 (3건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | parse | echo | over_b | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | ERROR(timeout after 180.0s) |  |  |  |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 | 48.0701 | 47.0487 | 380 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0537 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.5 | 1.0 | 8.6 | 41.5482 | 534 | 0 | 0 | 0 | 0 | 0 |  |  | 0.7738 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | None | 75.4981 | 44.2071 | 128 | 0 | 0 | 0 | 0 | 0 |  |  | 0.08 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | None | None | None | 51.9865 | 44.7156 | 821 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0573 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None | 8.5677 | 39.3675 | 544 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.8512 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | None | 7.5203 | 38.5007 | 555 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.8307 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | None | 6.7801 | 35.8683 | 284 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.7892 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
