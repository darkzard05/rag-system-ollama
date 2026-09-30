# Citation measurement — identical text on [1]-[4], distinct pages (Wave-0 Task 1)

- Mode: **STATIC-PARTIAL** (no live same-PDF repro claimed)
- Timestamp: 20260926_142832 · Query: `cm3` · Companion: `measure_citation_20260926_142832.json`
- Scope: read-only under `src/`; no fixes, no commits, no files outside `reports/` (+ OS temp unused; grouping port ran inline via `python -c`).

`VERDICT: branch D`

## 1. What was read (code path)

- `src/ui/components/chat_references.py` L60-175:
  - L79-91 fallback dedup: `key = (sec, page)` — with empty `citations`, one citation is minted **per (section, page)** pair; same-section chunks on distinct pages are never deduped.
  - L105-136 grouping: `page_to_refs` keyed by `int(target_p)` — exactly **one ref per page**; L112-130 normalizes generic sections (`""`, `"문서 본문"`, `"일반 본문"`, `"General content"`, `"none"`, `"?"`, plus `(p.N)` scrub) to `""`.
  - L144-171 labels: `f"{sec} · p.{page}"` or `f"p.{page}"` when section is `""` — **built from chunk metadata, not LLM output**.
- `src/api/schemas.py` `Citation{doc_id, text_span, section, page, score}` / `AnswerStructure` — structured citations ride a `citations` stream event (`_generate.py` L523-529) into `result["citations"]` (L566-571) and `streaming_core.py` aux state (L867-876).
- `src/core/document_hydrator.py` L81-121: join keys are **(file_hash, page_num)** via `coord_cache.get_coords_batch(file_hash, page_nums)`; `doc_stable_id()` = `metadata.doc_id` else `fast_hash(page_content)` (`src/common/utils.py` L53-73).
- `""` sections come from: streaming-path `meta.get("current_section", "")`, fallback default `"문서 본문"` → normalized to `""`, `format_context` default `"일반 본문"` → normalized to `""`.

## 2. Live-backend check (probe, not repro)

- `ollama list`: OK — `qwen3:4b-instruct-2507-q4_K_M`, `nomic-embed-text-v2-moe:latest` present.
- Streamlit app: **not running** (no streamlit process). Coord-cache DB present (`.model_cache/coord_cache/coords.db`). 457 PDFs under `data/temp`.
- `config.yml` has **no** `rag.prompt_templates` section (`version: null`, code default `"1.0"`; incident turn reported `prompt_version=2.0` per brief).
- Blocker: the incident turn's **source PDF / file_hash is unrecorded** — no session transcript maps query `cm3` → file (repo `cm3` hits are only tests/benchmarks, e.g. rerank bench on `2201.07520v1.pdf` which records timings only, no chunk texts). A fresh `cm3` run now would index an arbitrary PDF and produce different chunks — **not** the same-PDF repro — so none was attempted.

## 3. Deciding measurement

Known facts: 5 docs, lengths **405/346/183/581/219**, pages **[3,10,11,14,19]**, labels from chunk metadata, LLM citation-injection ruled out, `prompt_version=2.0`.

- **Branch H (identical hashes + distinct pages): EXCLUDED.** All five lengths are distinct → full texts cannot be byte-identical → content hashes cannot be identical. A hydration join-key fan-out of one text across pages contradicts the recorded lengths.
- **Branch E (distinct hashes + identical text): NOT SUPPORTED.** Distinct lengths imply distinct full texts; "identical text" can only refer to a short label/section string — a display artifact, not extraction duplication. No evidence of duplicated extraction; re-index not indicated.
- **Branch D (same-section-different-page minting with `""` sections): SUPPORTED.** Executed a verbatim port of L79-136 + L144-150 via python CLI on synthetic docs (empty/generic sections, the five pages): minted 5 fallback citations, `page_to_refs = {3,10,11,14,19}` all `section=""`, labels `p.3 / p.10 / p.11 / p.14 / p.19` — i.e. the renderer **by construction** emits near-identical refs differing only by page suffix. Reconstructed `page_to_refs` for [1]-[4]: `[1]→p.3, [2]→p.10, [3]→p.11, [4]→p.14` (`[5]→p.19`).
- Per-chunk `sha256` / `text_trim200` / `doc_stable_id` / `current_section` / `file_hash` are **null in the JSON by necessity** (raw texts unrecoverable statically); normalization scheme for the live rerun is prescribed there.

## 4. Renderer inputs (for [1]-[4])

- `documents`: 5 retrieved docs (lengths/pages above; length↔page pairing unrecorded — verdict is pairing-invariant).
- `citations`: fallback path assumed (absent/empty → minted via `(sec,page)` dedup); a structured `citations[]` input would still fan out per page through the same grouping.
- `page_to_refs` + labels: see §3 (full tables in JSON `renderer_inputs`).

## 5. Task-2 live rerun requirements (BONUS not captured — no live turn)

Record source PDF path + `file_hash` + session transcript; capture the **raw LLM completion string** and **final rendered body** here; fill `sha256_normalized` (strip → `/\s+/ → single space` → sha256/utf-8), `text_trim200`, `doc_stable_id`, `current_section`, `(file_hash, page)` keys; re-apply the decision rule to confirm or overturn branch D.

`VERDICT: branch D`
