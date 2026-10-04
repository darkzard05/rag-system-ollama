# Follow-up note for commit 2869337 (no history rewrite)

- Commit: `2869337442dc88dbf3a6457b4d5233a6d13eeb29`
  (`perf(retrieval): widen semantic routing keywords and rerank top_k pool`,
  2026-09-30).
- Commit message claims: the 12-question routing matrix in
  `reports/repro_h3_weights_20260927_110256.md` shows `0.0% misroute (0/12)`.
- Correction (F1 Task 4, BLEND_DECISION strict 3-way): the linked report
  excluded 4 dual-match PARTIAL rows from the denominator while its own S6
  finding admits a genuine misroute. Re-measured with the corrected generator
  (`scripts/evaluation/repro_h3_weights.py`, strict 3-way mirroring
  `src/core/graph/_retrieval.py` lines 70-84, denominator == row count):
  **Misrouted 4/12 = 33.3%** — K1/K5/K6/S6 fall through to the static default
  `[0.4, 0.6]`, which matches neither pure route. Zero `[0.6, 0.4]` blends are
  emitted (the code has no blend branch; ledger F1-9).
- The `0/12` figure was a measurement-loop artifact (simulated blend +
  denominator exclusion), not a code regression: the keyword-list widening in
  2869337 is retained; only the reported number is superseded by this note.
- Pushed history is left untouched; this file reconciles the record.
