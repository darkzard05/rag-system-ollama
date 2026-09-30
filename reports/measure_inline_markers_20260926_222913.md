# Inline [N] Marker Reliability — qwen3:4b-instruct-2507-q4_K_M (live, headless)

- Date: 2026-09-26 22:24–22:30 KST · Report: `reports/measure_inline_markers_20260926_222913.md`
- Scope: experiment-only. No `src/` / `config.yml` / `tests/` edits, no commits
  (evidence: `git diff --stat` empty; working tree shows only pre-existing untracked report files).
- Raw captures (OS temp, not committed):
  `exp_inline_markers.py` + `inline_markers_raw.json` (answers, scores, doc heads),
  `exp_inline_supp.py` + `inline_markers_docs.json` (retrieved-doc stable ids).

## 1. Setup

- `ollama list` confirmed both models present:
  `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) and `nomic-embed-text-v2-moe` (957 MB). Not LIVE-BLOCKED.
- Headless drive: `RAGSystem` imported from `src` (`build_pipeline` + `aquery`,
  session `exp-inline-markers`), PDF `tests/data/2201.07520v1.pdf` (CM3 paper).
  Fresh index build: 83 chunks, FAISS Flat. No Streamlit.
- In-memory-only prompt override (throwaway script, never written to disk):
  appended to the `structured_output` template copy
  (`PROMPT_TEMPLATES_CONFIG["structured_output"]`, identity-asserted against
  `core.graph._generate`'s reference):
  "The [Context] lists retrieved documents IN ORDER: first = [0], second = [1], …
  After EVERY claim sentence in `final_answer`, append the [N] marker of the
  0-based Nth retrieved document supporting that sentence. ONLY bare [N] markers."
  The standing "never insert `[doc:N]` tags" rule was suspended in-memory for this run.
- prompt_version=2.0 structured JSON output; server logs show JSON parse success on all 3 queries.
- Confound to keep in mind: `format_context` presents docs to the LLM as
  `[doc:<16-hex-stable-id>]` tokens (not `[0]`-style numbers), so the model had to
  map positions itself.

## 2. Per-query results (5 docs retrieved on every query; `relevant_docs_count` = 5/5/5)

| # | Query | [N] emitted? | Marker validity vs retrieved indices (0–4) | Misplaced / missing (strict) |
|---|-------|--------------|--------------------------------------------|------------------------------|
| Q1 | `cm3` | NO — 0× `[N]`; emitted 3 hash tokens `[6daa326dee4d224d] [c41db7e13ee8181f] [8bb3eece705010b2]` | N/A (no [N]); the 3 hashes resolve to genuinely retrieved docs (positions 0, 1, 3 per re-retrieval run) — right docs, wrong scheme | 3/3 claim sentences lack any marker → missing=3 |
| Q2 | causally-masked LM vs standard MLM | YES — 5× `[N]`, dumped en bloc as `[0][1][2][3][4]` after the last sentence | 5/5 range-valid (all < 5 docs) | 5/5 misplaced: blob attaches to no claim sentence; 3/3 claim sentences lack markers → missing=3; blanket-cites all docs with zero discrimination |
| Q3 | image tokenization + zero-shot tasks | NO — 0× `[N]`; emitted `[doc:df6864ec5e2c1dc8] [doc:8bb3eece705010b2]` | N/A (no [N]); both ids resolve to retrieved docs (doc[1], doc[3] per re-retrieval run) — right docs, copied `[doc:…]` scheme | 2/2 claim sentences lack markers → missing=2 |

Strict scoring rules applied: a wrong-scheme token is a failure (not partial credit);
a range-valid number dumped away from its claim sentence is misplaced, not correct;
attachment requires the marker on the claim sentence it supports.

## 3. Raw completions (verbatim `final_answer`)

Q1 (`cm3`, 12.8 s):
> CM3는 이미지와 텍스트를 포함한 전체 문서 구조를 모델링하는 원시적 다중 모달 모델로, 문장 단위의 생성을 위해 원인적 마스킹을 사용하며, 언어 모델로서의 기능도 수행할 수 있습니다. 이 모델은 다양한 언어 작업에 대해 영향 없이 성능을 발휘하며, 이미지 생성, 이미지 설명, 무조건적 이미지 생성 등 다양한 기능을 프롬프트로 복원할 수 있습니다. CM3는 Aghajanyan et al. (2021)의 BART 유형의 손실 함수에서 벗어나, 원인적 마스킹을 사용한 새로운 손실 함수를 도입했습니다. [6daa326dee4d224d] [c41db7e13ee8181f] [8bb3eece705010b2]

Q2 (22.7 s):
> CM3의 causally masked language modeling은 토큰을 왼쪽에서 오른쪽으로 생성하면서, 일부 긴 토큰 스피언을 원래 위치에서 마스킹하고 그 후 문자열 끝에서 생성하는 방식으로, 일반적인 마스킹 언어 모델과 달리 원래 위치를 유지하지 않고 끝에서 생성합니다. 이는 완전한 생성 모델링과 양방향 컨텍스트를 동시에 제공하는 혼합 모델을 가능하게 합니다. 이 방법은 DALL-E와 달리 이미지의 연속된 섹션을 주변 컨텍스트에 기반하여 채우는 데 사용될 수 있습니다. [0][1][2][3][4]

Q3 (12.0 s):
> CM3는 이미지를 연속된 토큰으로 표현하며, 이는 주변 맥락에 기반한 이미지 채우기(인프일링)를 가능하게 합니다. 문서는 CM3가 이미지 생성, 이미지 캡션, 무조건적 이미지 생성 등 다양한 영역에서 제로샷 성능을 보임을 보고합니다. [doc:df6864ec5e2c1dc8] [doc:8bb3eece705010b2]

Retrieved-doc heads per query are stored in the raw JSON (temp). Spot-check notes:
Q1 claims (BART-loss departure, capability list) are topically covered by retrieved
docs 0/1/3, but no sentence carries its marker. Q2's DALL-E sentence is supported by
retrieved docs 2/3 only, yet the blob cites all five — indiscriminate. Q3's two
sentences are covered by docs 1/3/4, but markers use the unparseable-by-spec scheme.

## 4. Failure-mode analysis

1. **Scheme-copying (Q1, Q3: 2/3 queries).** The model ignores the 0-based `[N]`
   instruction and echoes the context's native `[doc:<stable-id>]` format instead —
   twice with full `doc:` prefix, once as bare hashes. Provenance tracking works
   (all 5 echoed ids resolve to actually-retrieved docs, verified by an independent
   re-retrieval run in session `exp-inline-markers-2`), but the instructed `[N]`
   scheme is not followed, so a `[N]`-expecting parser would extract zero citations.
2. **En-bloc dump (Q2: 1/3 queries).** The model can emit range-valid `[N]`
   numbers (5/5 in range) but clusters them after the final sentence instead of
   trailing each claim — a parser attributing `[N]` to its host sentence would
   attach all five to an empty blob and leave every claim uncited.
3. Scoreboard: claim sentences with a correct sentence-trailing valid `[N]`: **0/8**.
   Queries with ≥1 correctly placed valid marker: **0/3**. Clean `[N]` compliance: 0/3.

## 5. Limitations

- Claim-attachment auto-check (content-token overlap, ≥2 shared tokens of length ≥5)
  plus manual spot-check; Korean paraphrase could hide support, but placement failures
  (missing/en-bloc/wrong-scheme) decide the verdict before attachment semantics matter.
- Hash→doc-position mapping rests on a same-PDF re-retrieval run (retrieval is
  deterministic: Flat FAISS + BM25 + FlashRank over identical chunks); run-1 doc-head
  texts are consistent with the mapped positions.
- `aquery` does not surface `parse_failed`; JSON-parse success taken from server logs.
- n=3 by design; enough to reject adoption, not to certify it.

## 6. Verdict

`VERDICT: defer`

The 4b model tracks provenance (echoed ids hit real retrieved docs) but does not
reliably emit sentence-trailing `[N]` markers: 0/3 queries compliant, 0/8 claim
sentences correctly marked. Building a prompt+parser feature on `[N]` emission is not
justified. If revisited: either number the context docs `[0]`-style in the template
(removing the positional-mapping burden and the competing hash scheme), or write the
parser against the model's preferred `[doc:<stable-id>]` echoes — then re-run this
same 3-query protocol before adopting.
