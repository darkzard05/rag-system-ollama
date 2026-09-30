# H5 Brevity + Query-Cache Repro

- Timestamp: 20260927_091923 | No Ollama, no src/ edits (EXTERNAL repro)
- config: prompts [Brevity] band 400~600 chars; query_cache.enabled=false, similarity_threshold=0.6, ttl=3600s, min_confidence=0.85
- test data: tests/data/golden_set.json (5 queries), testset_2201.csv headers: question,ground_truth,reference_context

## (a) Answer-length vs completeness (ANALYTICAL, from reports/*.json)

- report files with answer_chars: 100 question rows (24 non-empty answers, 76 empty/no-LLM rows)
- brevity band [400,600]: in-band=9 (38%), below=9, above=6
- mean answer_chars (non-empty): 424.12 | max=834.0 min=16.0
- mean context_recall (stored retrieval field): 0.56 (n=6)
- Tradeoff reading: the 400~600-char band caps verbosity but past eval rows show most no-LLM runs store answer_chars=0 (retrieval-only), so completeness must be judged via context_recall/rerank scores, not length. Enforcing the band at generation time risks cutting citations on multi-hop questions (H5 hypothesis: band OK for factoid, tight for comparison/synthesis).

length histogram (non-empty answers):
- [0,400] below-band: ▓▓▓▓▓▓▓▓▓ (9)
- [400,600] in-band: ▓▓▓▓▓▓▓▓▓ (9)
- [600,1000] over-band: ▓▓▓▓▓▓ (6)
- [1000,inf] long:  (0)

## (b) Query-cache hit-rate (SIMULATED token-overlap — NOT embedding cosine)

| # | A | B | kind | sim | >=0.88 | >=0.92 | >=0.95 |
|---|---|---|---|---|---|---|---|
| 1 | CM3 모델의 핵심 기여 3가지를 요약해줘 | CM3 모델의 주요 기여 3가지를 요약해줘 | near-dup | 0.7071 | - | - | - |
| 2 | CM3 모델의 핵심 기여 3가지를 요약해줘 | 이 논문의 핵심 기여점 3가지만 요약해줘 | paraphrase | 0.2875 | - | - | - |
| 3 | 트랜스포머 구조가 어떻게 작동하는지 설명해줘 | 트랜스포머 구조의 작동 방식을 설명해줘 | near-dup | 0.325 | - | - | - |
| 4 | 트랜스포머 구조가 어떻게 작동하는지 설명해줘 | 어텐션 메커니즘의 원리를 알려줘 | paraphrase | 0.0 | - | - | - |
| 5 | BERT와 GPT의 차이점은 무엇인가요? | BERT와 GPT의 차이를 설명해줘 | near-dup | 0.3841 | - | - | - |
| 6 | BERT와 GPT의 차이점은 무엇인가요? | 두 모델의 설계 철학 차이는 무엇인가요? | paraphrase | 0.1627 | - | - | - |
| 7 | 이미지 생성 모델의 최신 트렌드를 알려줘 | 이미지 생성 모델의 최근 동향을 알려줘 | near-dup | 0.4881 | - | - | - |
| 8 | 이미지 생성 모델의 최신 트렌드를 알려줘 | DALL-E 이후 이미지 모델 발전 과정을 요약해줘 | paraphrase | 0.0858 | - | - | - |
| 9 | v2.1 버전의 변경 사항은? | v2.1 버전에서 바뀐 점은? | near-dup | 0.2381 | - | - | - |
| 10 | v2.1 버전의 변경 사항은? | 최신 릴리스의 주요 업데이트 내용은? | paraphrase | 0.0192 | - | - | - |
| 11 | 0x1F4A 레지스터의 역할은? | 0x1F4A 주소 레지스터는 무슨 역할을 하나? | near-dup | 0.2425 | - | - | - |
| 12 | 0x1F4A 레지스터의 역할은? | 하드웨어 레지스터 매핑 방식을 설명해줘 | paraphrase | 0.0556 | - | - | - |
| 13 | 문서에서 언급된 파라미터 수는? | 문서에 나온 모델 파라미터 규모는? | near-dup | 0.2054 | - | - | - |
| 14 | 문서에서 언급된 파라미터 수는? | 학습 데이터셋의 규모는 어떻게 돼? | paraphrase | 0.0 | - | - | - |
| 15 | 안녕, 너는 누구니? | 안녕, 자기소개 해줘 | near-dup | 0.1714 | - | - | - |
| 16 | 안녕, 너는 누구니? | CM3 이미지 토큰화 방식을 설명해줘 | unrelated | 0.0 | - | - | - |
| 17 | 요약해줘 | 요약해 줘 | near-dup | 0.5 | - | - | - |
| 18 | 요약해줘 | 전체 문서의 결론을 3줄로 정리해줘 | paraphrase | 0.0312 | - | - | - |
| 19 | 지도학습과 비지도학습의 장단점 비교 | 지도학습과 비지도학습의 장단점을 비교해줘 | near-dup | 0.5104 | - | - | - |
| 20 | 지도학습과 비지도학습의 장단점 비교 | 강화학습의 기본 개념을 설명해줘 | unrelated | 0.0435 | - | - | - |

Threshold summary (SIMULATED):

- thr=0.88: hits 0/20 (0%) | near-dup recall 0/10 | unrelated false-hits 0/2
- thr=0.92: hits 0/20 (0%) | near-dup recall 0/10 | unrelated false-hits 0/2
- thr=0.95: hits 0/20 (0%) | near-dup recall 0/10 | unrelated false-hits 0/2
- Note: token-overlap UNDER-scores true paraphrases vs embedding cosine; config default 0.92 would hit near-dups but miss most paraphrases here. Real hit-rate needs the embedder (Ollama) — out of scope for this repro.

## (c) Dead-path ref check

- src/core/graph/_retrieval.py:98 -> LIVE cache call-site ✅ (`                cache_key = f"{file_hash}:{query}"`)
- src/core/graph/_generate.py:640 -> LIVE cache call-site ✅ (`                    query_for_cache = f"{file_hash}:{query_for_cache}"`)
- Verdict: both refs are LIVE (query-cache get in _retrieval.py, set in _generate.py), gated by QUERY_CACHE_ENABLED (config enabled=false) — no dead path; the feature is dormant opt-in, not dead code.

_All similarity numbers in (b) are SIMULATED (token overlap). No model calls made._
