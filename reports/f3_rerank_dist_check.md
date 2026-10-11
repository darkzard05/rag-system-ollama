# F3 grade short-circuit distribution (numbers-first gate)

- corpus: `eval_quality_top3verify_20261004_104540.json`
- meta: tag=top3verify ts=2026-10-04T10:47:12 no_llm=True
- active rerank engine (this process): `flashrank`
- grading config: min_score_to_skip=0.7 semantic=0.45 gap=0.05 top_k=10 grade_top_n=5
- judge-valid rows: 0/15 (mid-batch None->score backfills excluded; see note)

## 1. max_rerank distribution (engine-scale split)

Scale label is heuristic (>0.60 = flashrank-like; eval JSON records no engine field; ranges per _grade_verify.py:235).

| query # | max_rerank | scale | n_docs |
| --- | --- | --- | --- |
| 1 | 0.8838 | flashrank-like | 5 |
| 2 | 0.0537 | semantic-like | 5 |
| 3 | 0.7738 | flashrank-like | 5 |
| 4 | 0.0000 | semantic-like | 0 |
| 5 | 0.0573 | semantic-like | 5 |
| 6 | 0.8512 | flashrank-like | 5 |
| 7 | 0.8307 | flashrank-like | 5 |
| 8 | 0.7892 | flashrank-like | 5 |
| 9 | 0.8948 | flashrank-like | 5 |
| 10 | 0.0740 | semantic-like | 5 |
| 11 | 0.9600 | flashrank-like | 5 |
| 12 | 0.0637 | semantic-like | 5 |
| 13 | 0.7800 | flashrank-like | 5 |
| 14 | 0.0605 | semantic-like | 5 |
| 15 | 0.0578 | semantic-like | 5 |

| max_rerank bin | count |
| --- | --- |
| <0.10 | 7 |
| 0.10-0.45 | 0 |
| 0.45-0.70 | 0 |
| 0.70-0.85 | 4 |
| >=0.85 | 4 |

## 2. branch-fire counts (exact _grade_verify.py:256 logic)

| threshold | fires / n | fraction |
| --- | --- | --- |
| 0.85 | 4/15 | 0.267 |
| 0.70 | 8/15 | 0.533 |
| 0.45 | 8/15 | 0.533 |

## 3. rerank log sample (recorded ranks) + rank-drop table

Eval logs record only the top-5 retrieved docs per query (prod pool is top_k=10 -> grade_top_n=3); ranks below are within the recorded slice.

- Q1 ranks: 0.8838, 0.8681, 0.8145, 0.6064, 0.3023
- Q2 ranks: 0.0537, 0.0485, 0.0468, 0.0421, 0.0412
- Q3 ranks: 0.7738, 0.5742, 0.4061, 0.0953, 0.0598

| query # | cited scores -> rank | lossy (rank>5) |
| --- | --- | --- |
| 1 | (no citations) | - |
| 2 | (no citations) | - |
| 3 | (no citations) | - |
| 4 | (no citations) | - |
| 5 | (no citations) | - |
| 6 | (no citations) | - |
| 7 | (no citations) | - |
| 8 | (no citations) | - |
| 9 | (no citations) | - |
| 10 | (no citations) | - |
| 11 | (no citations) | - |
| 12 | (no citations) | - |
| 13 | (no citations) | - |
| 14 | (no citations) | - |
| 15 | (no citations) | - |

## 4. TTFT: totals by branch outcome + stage-split note

| outcome @0.70 | n | mean TTFT(s) |
| --- | --- | --- |
| fires (short-circuit path) | 0 | 0.00 |
| no-fire (full LLM grade path) | 0 | 0.00 |

NOTE (stage split): scripts/evaluation/eval_quality.py records TTFT totals only — no retrieve(k)/rerank/grade stage breakdown exists in the schema. A per-stage retrieve-vs-rerank-vs-grade(LLM-vs-shortcircuit) split needs an instrumented live run and is out of scope for this distribution gate; the by-outcome proxy above stands in for it.

## 5. engine-scale comparison (recorded corpus)

- compare: `eval_quality_baseline_20260806_174429.json` (semantic/bi-encoder scale)
| threshold | latest fires | compare fires |
| --- | --- | --- |
| 0.85 | 4/15 | 0/8 |
| 0.70 | 8/15 | 0/8 |
| 0.45 | 8/15 | 1/8 |

## 6. verdict (numbers only)

- README documents min_score_to_skip=0.85; config sets 0.70 (semantic 0.45); code fallbacks are 0.45 general / 0.40 semantic / gap 0.03 — triple inconsistent.
- Fire counts above are computed with the exact max>=threshold AND (gap OR both-high) predicate; 0.70 is unreachable for semantic-scale scores (cosine cap ~0.57 per config.yml:262).
- Task 6 fix is gated on these numbers; no thresholds were touched by this script.
