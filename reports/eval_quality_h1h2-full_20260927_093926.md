# RAG Answer Quality Evaluation Report

- Date: 2026-09-27T09:44:52
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: False / testset_n: 3

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 4 |
| P@1 | 0.0 |
| MRR@5 | 0.1875 |
| context_recall (golden) | 0.4444 (3건) |
| latency_question_count | 7 |
| avg TTFT (s) | 19.5683 |
| avg TPS | 37.3004 |
| avg answer chars | 1312.2857 |
| avg eval_count | 486.0 |
| truncated ratio | 0.0 (0건) |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | None (0건) |
| faithfulness avg (1-5) | None (0건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.25 | 0.5 | 7.5457 | 47.7845 | 705 | 0 | 0 |  |  | 0.9012 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 | 50.5512 | 40.7912 | 442 | 0 | 0 |  |  | 0.0522 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 | 0.3333 | 9.8149 | 33.4291 | 775 | 0 | 0 |  |  | 0.8776 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | None | 62.5079 | 40.0098 | 27 | 0 | 0 |  |  | 0.0787 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | 0 | 0.5 | None | 46.8878 | 35.0304 | 493 | 0 | 0 |  |  | 0.0609 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None | 8.8045 | 33.5893 | 450 | 0 | 0 |  |  | 0.8512 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | None | 8.3105 | 35.7769 | 286 | 0 | 0 |  |  | 0.8359 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | None | 5.0634 | 34.7015 | 251 | 0 | 0 |  |  | 0.8048 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
