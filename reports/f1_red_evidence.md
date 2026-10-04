# F1 Weight Measurement-Loop Defect — RED Evidence

- Test: `tests/unit/test_f1_weight_measurement_red.py` (TDD RED, must FAIL)
- Command: `python -m pytest tests/unit/test_f1_weight_measurement_red.py -q`
- Result: **2 failed, 2 passed** (blend + misroute RED; pure-type controls GREEN)
- Style: `ruff check` clean, `ruff format --check` clean, type annotations on all tests
- Real path: `preprocess` from `core.graph.graph_builder` (= `src/core/graph/_retrieval.py`
  lines 70-84) with real `DYNAMIC_WEIGHTING_CONFIG` / `ENSEMBLE_WEIGHTS`; no mock
  of the weight decision. Ledger: `reports/_plan_evidence_ledger.md` F1 (F1-1..F1-10)
  + Git section (commit 2869337).

## pytest output (verbatim)

```text
FAILED tests/unit/test_f1_weight_measurement_red.py::test_report_blend_rows_reproducible
FAILED tests/unit/test_f1_weight_measurement_red.py::test_misroute_zero_of_twelve_reproducible
2 failed, 2 passed in 2.33s
```

(a) `AssertionError: K1: report claims [0.6, 0.4] but real path emits
{'bm25': 0.4, 'faiss': 0.6}` — `assert [0.4, 0.6] == [0.6, 0.4]`, at index 0
diff: 0.4 != 0.6.

(b) `AssertionError: PARTIAL-included misroute is 4/12, not report 0/12` —
`assert 4 == 0`.

Controls PASS: pure keyword (`BERTBase123 ...`) -> `{"bm25": 0.8, "faiss": 0.2}`;
pure semantic (`두 모델의 설계 철학 차이는 무엇인가요?`) -> `{"bm25": 0.2, "faiss": 0.8}`.

## Code-vs-report diff excerpt

Code (`src/core/graph/_retrieval.py:77-83`, ledger F1-2..F1-7):

```python
if is_keyword_heavy and not is_semantic_heavy:
    kw_w = DYNAMIC_WEIGHTING_CONFIG.get("keyword_weight", 0.8)  # F1-3
    weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}      # F1-4
elif is_semantic_heavy and not is_keyword_heavy:
    sm_w = DYNAMIC_WEIGHTING_CONFIG.get("semantic_weight", 0.8)  # F1-6
    weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}      # F1-7
```

Emittable set: `[0.4, 0.6]` default (F1-10, `config.yml:90`), `[0.8, 0.2]`
keyword-only, `[0.2, 0.8]` semantic-only; both-match falls back to default.
`grep 0.6` in `_retrieval.py`: no hits in the dynamic path (F1-9).

Report (`reports/repro_h3_weights_20260927_110256.md`) claims 4 rows at
`[0.6, 0.4]` (K1/K5/K6/S6) — a blend the code cannot emit; the real path
returns `[0.4, 0.6]` for all four (dual keyword+semantic match -> fallback).

Report claims `Misrouted: 0/12` with PARTIAL rows excluded, while its own
Finding (S6, line 36) admits a "Single genuine misroute"; commit 2869337
message repeats `0.0% misroute (0/12)`. Real path with PARTIAL included:
**4/12** (K1/K5/K6/S6 fall back to `[0.4, 0.6]`, matching neither pure route).

## Decision (F1 Task 4 — BLEND_DECISION: strict 3-way, report-side fix)

- **Chosen: keep strict 3-way; correct the report generator to never emit
  `[0.6, 0.4]`** (preferred minimal path — the code has no blend branch).
- **Rejected: implementing a real dual-match `[0.6, 0.4]` blend branch.**
  It would change routing behavior (scope creep into Task 5/F2 tie-break,
  which owns dual-match handling) and contradict the F2 RED baseline that
  assumes silent `[0.4, 0.6]` fallthrough.
- **Rationale:** the defect is a measurement loop — the old generator
  (`scripts/evaluation/repro_h3_weights.py`) simulated a blend the real
  `preprocess` path (`src/core/graph/_retrieval.py` lines 70-84, guards
  mutually exclusive, `grep 0.6` no hits per F1-9) can never emit, and it
  excluded PARTIAL rows from the denominator. The failing RED asserts encode
  **code-side routing** (`[0.6, 0.4]` must reproduce via the real path) and
  report-side-corrected numbers, so the fix belongs **report-side**; the
  weight code is unchanged except for this decision comment. Task 5 (F2
  tie-break) follows this decision.
- **Fix applied:** `classify()` both/neither fall through to `[0.4, 0.6]`;
  heatmap bins reduced to the code-enumerated set
  `{[0.8,0.2],[0.4,0.6],[0.2,0.8]}`; verdict counts every row
  (`Misrouted = misrouted/total`, denominator == 12 → **4/12 = 33.3%**).
- **Record reconciliation:** `reports/f1_2869337_followup_note.md` supersedes
  the `0/12` claim (no history rewrite). CI self-check:
  `scripts/evaluation/check_f1_report.py` (weights ∈ code-enumerated set,
  denominator == row count).
