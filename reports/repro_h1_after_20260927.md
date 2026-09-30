# H1 Strict-JSON repro report

- Timestamp (UTC): 2026-09-27T00:25:37Z
- Config: `C:\Users\darkzard05\hy\rag-system-ollama\config.yml`
- prompt_templates.version: `2.0` (expected `2.0`)
- Rules present: `{"rule10_strict_escaping": true, "rule11_no_repetition": true, "ssot_citations_only": true, "no_inline_doc_tag_rule": true, "json_schema_present": true}`
- Refs (read-only):
  - `config.yml` prompt_templates structured_output v2.0 (§263-333)
  - `src/core/graph/_generate.py` format_context + json.loads + _repair_json fallback
  - `scripts/evaluation/eval_quality.py` SCORE_PROMPT (faithfulness judge)

## Metrics

- parse-failure rate: **5/7 = 71.4%**
- echo-case rate (5-gram repeat > 0): **1/7 = 14.3%**
- SSoT-violation rate (inline `[doc:N]` in final_answer): **1/7 = 14.3%**
- elapsed: 0.04s (budget <60s)

## Per-case results

| case | strict parse | 5-gram echo | SSoT | note |
|---|---|---|---|---|
| `Q1-quotes-causal-masking` | FAIL: JSONDecodeError: Expecting ',' delimiter: line 1 column 25 (char 24) | 0/0 (0.00) | ok | raw double-quotes inside value (rule 10 violation) |
| `Q2-version-numbers` | FAIL: JSONDecodeError: Expecting ',' delimiter: line 1 column 44 (char 43) | 0/3 (0.00) | ok | quotes + parens + version/date tokens |
| `Q3-raw-newline` | FAIL: JSONDecodeError: Invalid control character at: line 1 column 26 (char 25) | 0/2 (0.00) | ok | literal newline inside JSON string value |
| `Q4-echo-repeat` | OK | 14/14 (1.00) | ok | repeated fact; echo demo repeats claim 3x (rule 11 violation) |
| `Q5-nested-quotes-ko-en` | FAIL: JSONDecodeError: Expecting ',' delimiter: line 1 column 24 (char 23) | 0/3 (0.00) | ok | multiple raw quote pairs |
| `Q6-tab-backslash` | FAIL: JSONDecodeError: Invalid control character at: line 1 column 22 (char 21) | 0/0 (0.00) | ok | raw tab + backslashes inside value |
| `Q7-docid-ssot` | OK | 0/2 (0.00) | VIOLATION | final_answer embeds inline [doc:0] tag -> SSoT violation |

## Interpretation

Naive small-LLM emission copies adversarial spans verbatim (no rule-10 escaping), so strict `json.loads` fails on raw `"` / literal newline / tab cases — the H1 Strict-JSON failure. `_repair_json` in `src/core/graph/_graph_core.py` exists as a fallback, but the repro shows the raw emission itself violates v2.0 §10 before repair. Echo (rule 11) is demonstrated by Q4 (same 5-gram repeated 3x); SSoT violation by Q7 (inline `[doc:N]` inside `final_answer` instead of citations[] only).

## Repro command

```powershell
python scripts/evaluation/repro_h1_json.py --report reports/repro_h1_<ts>.md
```
