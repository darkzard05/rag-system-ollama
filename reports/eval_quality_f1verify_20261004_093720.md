# RAG Answer Quality Evaluation Report

- Date: 2026-10-04T09:39:16
- PDF: tests/data/2201.07520v1.pdf
- Model: qwen3:4b-instruct-2507-q4_K_M / Embedder: nomic-embed-text-v2-moe
- num_predict: 2048 / num_ctx: 8192
- Truncation threshold: 1843.2 (0.9 x num_predict)
- --no-llm: True / testset_n: 12

## Aggregates

| Metric | Value |
| --- | --- |
| scorable_count (golden, 퇴화 제외) | 3 |
| P@1 | 0.0 |
| MRR@5 | 0.2333 |
| context_recall (golden) | 0.6667 (3건) |
| latency_question_count | 14 |
| avg TTFT (s) | None |
| avg TPS | None |
| avg answer chars | 0.0 |
| avg eval_count | None |
| truncated ratio | None (0건) |
| parse_failed / echo / over_budget | 0 / 0 / 0건 |
| overflow ratio | 0.0 (0건) |
| judge avg (1-5) | None (0건) |
| faithfulness avg (1-5) | None (0건) |
| eval_count missing | 14건 |

## Per-Question

| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | parse | echo | over_b | judge | faithfulness | max_rerank | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | 1 | CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤  | 포함 | 0 | 0.2 | 0.5 |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.9012 |  |
| golden | 2 | 문서 내에서 언급된 모델의 파라미터 수와 학습 데이터셋의 규모는 어떻게 돼? | 포함 | 0 | 0.0 | 0.5 |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.0522 |  |
| golden | 3 | 이 논문의 핵심 기여점(Contribution) 3가지만 요약해줘. | 포함 | 0 | 0.5 | 1.0 |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.7738 |  |
| golden | 4 | 안녕, 너는 누구니? | 제외됨 |  |  |  |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| golden | 5 | 이미지 생성 모델의 최신 트렌드에 대해 알려줘. | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.0609 | out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대) |
| testset | 1 | CM3 모델은 어떤 종류의 모델을 기반으로 하고 있으며, 그 핵심 특징은 무엇인가요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.8512 |  |
| testset | 2 | CM3 모델이 훈련된 데이터 세트는 어떤 내용을 포함하며, 그 데이터의 순서는 어떻게 되나요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.8005 |  |
| testset | 3 | CM3 모델은 어떤 데이터를 기반으로 훈련되었나요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.8088 |  |
| testset | 4 | CM3 모델의 훈련에 사용된 목적 함수는 무엇인가요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.8948 |  |
| testset | 5 | Monte Melkonian의 정치적 성향과 정체성은 무엇인가요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.074 |  |
| testset | 6 | CM3 모델이 Aghajanyan et al. (2021)의 작업에서 어떻게 발전했나요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.959 |  |
| testset | 7 | 실험을 위해 각 데이터 소스에서 테스트 세트를 어떻게 생성했나요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.0624 |  |
| testset | 8 | 사이즈 힌트란 무엇이며, CM3에서 이 힌트의 사용이 어떤 영향을 미쳤나요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.7767 |  |
| testset | 9 | 모델 훈련에 사용된 모델의 크기는 무엇인가요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.0605 |  |
| testset | 10 | 모델 훈련에 사용된 하이퍼파라미터와 최적화 방법은 무엇인가요? | 포함 | None | None | None |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  | 0.0578 |  |

## Degenerate Questions

| row | reason |
| --- | --- |
| 4 | 빈 엔티티 (expected_key_entities=[]) |
