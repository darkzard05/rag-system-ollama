# F3 grade short-circuit distribution (numbers-first gate)

- corpus: `eval_quality_final-now_20260927_120511.json`
- meta: tag=final-now ts=2026-09-27T12:09:02 no_llm=False
- active rerank engine (this process): `flashrank`
- grading config: min_score_to_skip=0.7 semantic=0.45 gap=0.05 top_k=10 grade_top_n=3
- judge-valid rows: 3/8 (mid-batch None->score backfills excluded; see note)

## 1. max_rerank distribution (engine-scale split)

Scale label is heuristic (>0.60 = flashrank-like; eval JSON records no engine field; ranges per _grade_verify.py:235).

| query # | max_rerank | scale | n_docs |
| --- | --- | --- | --- |
| 1 | 0.9367 | flashrank-like | 5 |
| 2 | 0.0604 | semantic-like | 5 |
| 3 | 0.8777 | flashrank-like | 5 |
| 4 | 0.9116 | flashrank-like | 5 |
| 5 | 0.0651 | semantic-like | 5 |
| 6 | 0.8512 | flashrank-like | 5 |
| 7 | 0.8690 | flashrank-like | 5 |
| 8 | 0.8101 | flashrank-like | 5 |

| max_rerank bin | count |
| --- | --- |
| <0.10 | 2 |
| 0.10-0.45 | 0 |
| 0.45-0.70 | 0 |
| 0.70-0.85 | 1 |
| >=0.85 | 5 |

## 2. branch-fire counts (exact _grade_verify.py:256 logic)

| threshold | fires / n | fraction |
| --- | --- | --- |
| 0.85 | 5/8 | 0.625 |
| 0.70 | 6/8 | 0.750 |
| 0.45 | 6/8 | 0.750 |

## 3. rerank log sample (recorded ranks) + rank-drop table

Eval logs record only the top-5 retrieved docs per query (prod pool is top_k=10 -> grade_top_n=3); ranks below are within the recorded slice.

- Q1 ranks: 0.9367, 0.9240, 0.9012, 0.8719, 0.8715
- Q2 ranks: 0.0604, 0.0522, 0.0512, 0.0476, 0.0473
- Q3 ranks: 0.8777, 0.8776, 0.7738, 0.7083, 0.5742

| query # | cited scores -> rank | lossy (rank>3) |
| --- | --- | --- |
| 1 | (no citations) | - |
| 2 | (no citations) | - |
| 3 | 0.574->r5, 0.574->r5, 0.878->r2, 0.708->r4 | YES |
| 4 | (no citations) | - |
| 5 | (no citations) | - |
| 6 | 0.784->r4, 0.784->r4, 0.806->r3, 0.806->r3 | YES |
| 7 | (no citations) | - |
| 8 | 0.742->r4 | YES |

## 4. TTFT: totals by branch outcome + stage-split note

| outcome @0.70 | n | mean TTFT(s) |
| --- | --- | --- |
| fires (short-circuit path) | 6 | 11.76 |
| no-fire (full LLM grade path) | 2 | 36.06 |

NOTE (stage split): scripts/evaluation/eval_quality.py records TTFT totals only — no retrieve(k)/rerank/grade stage breakdown exists in the schema. A per-stage retrieve-vs-rerank-vs-grade(LLM-vs-shortcircuit) split needs an instrumented live run and is out of scope for this distribution gate; the by-outcome proxy above stands in for it.

## 5. engine-scale comparison (recorded corpus)

- compare: `eval_quality_baseline_20260806_174429.json` (semantic/bi-encoder scale)
| threshold | latest fires | compare fires |
| --- | --- | --- |
| 0.85 | 5/8 | 0/8 |
| 0.70 | 6/8 | 0/8 |
| 0.45 | 6/8 | 1/8 |

## 6. verdict (numbers only)

- README documents min_score_to_skip=0.85; config sets 0.70 (semantic 0.45); code fallbacks are 0.45 general / 0.40 semantic / gap 0.03 — triple inconsistent.
- Fire counts above are computed with the exact max>=threshold AND (gap OR both-high) predicate; 0.70 is unreachable for semantic-scale scores (cosine cap ~0.57 per config.yml:262).
- Task 6 fix is gated on these numbers; no thresholds were touched by this script.
