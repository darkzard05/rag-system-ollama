# RAG Answer Quality Evaluation Report

- Date: 2026-09-27T11:20:15
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: False / testset_n: 1

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 3 |
| P@1 | 0.0 |
| MRR@5 | 0.1111 |
| context_recall (golden) | 0.5556 (3건) |
| latency_question_count | 5 |
| avg TTFT (s) | 21.1559 |
| avg TPS | 52.0225 |
| avg answer chars | 1178.4 |
| avg eval_count | 407.0 |
| truncated ratio | 0.0 (0건) |
| parse_failed / echo / over_budget | 0 / 1 / 0건 |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | 4.0 (1건) |
| faithfulness avg (1-5) | 4.0 (1건) |
| eval_count missing | 0건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | parse | echo | over_b | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.3333 | 0.5 | 5.873 | 56.3411 | 459 | 0 | 0 | 0 | 0 | 0 |  |  | 0.9367 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 | 39.8361 | 52.972 | 79 | 0 | 0 | 0 | 0 | 0 |  |  | 0.0604 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.0 | 0.6667 | 6.7899 | 47.6022 | 540 | 0 | 0 | 0 | 0 | 0 |  |  | 0.8777 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 | None | None | None | 42.099 | 46.2 | 156 | 0 | 0 | 0 | 0 | 0 |  |  | 0.9116 |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | None | None | None | 46.8816 | 55.1531 | 162 | 0 | 0 | 0 | 1 | 0 |  |  | 0.0651 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None | 6.3991 | 48.0441 | 795 | 0 | 0 | 0 | 0 | 0 | 4 | 4 | 0.8512 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
