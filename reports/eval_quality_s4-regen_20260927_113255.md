# RAG Answer Quality Evaluation Report

- Date: 2026-09-27T11:37:07
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: False / testset_n: 3

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 3 |
| P@1 | 0.0 |
| MRR@5 | 0.1111 |
| context_recall (golden) | 0.5556 (3건) |
| latency_question_count | 7 |
| avg TTFT (s) | 16.2792 |
| avg TPS | 36.0234 |
| avg answer chars | 852.5714 |
| avg eval_count | 307.7143 |
| truncated ratio | 0.0 (0건) |
| parse_failed / echo / over_budget | 0 / 1 / 0건 |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 2.6667 (3건) |
| faithfulness avg (1-5) | 2.6667 (3건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | parse | echo | over_b | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.3333 | 0.5 | 5.5129 | 48.4409 | 370 | 0 | 0 | 0 | 0 | 0 |  |  | 0.9367 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 | 41.4967 | 47.4753 | 80 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0604 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 | 0.6667 | 7.039 | 39.3068 | 575 | 0 | 0 | 0 | 0 | 0 |  |  | 0.8777 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | None | 34.8305 | 41.7844 | 62 | 0 | 0 | 0 | 0 | 0 |  |  | 0.9116 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | None | None | None | 39.3058 | 13.7154 | 139 | 0 | 0 | 0 | 1 | 0 |  |  | 0.0651 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None | 7.6735 | 34.1173 | 709 | 0 | 0 | 0 | 0 | 0 | 4 | 4 | 0.8512 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | None | 7.5222 | 35.0877 | 47 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.869 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | None | 5.404 | 34.0205 | 234 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.8101 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
