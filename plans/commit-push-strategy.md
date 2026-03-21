# Plan: Commit and Push Changes

## Objective
Commit and push the changes made to optimize the RAG system for Korean language support and improve overall retrieval performance.

## Changes to Commit
1.  **`config.yml`**:
    -   Updated `default_embedding` to `nomic-embed-text-v2-moe` (multilingual support).
    -   Updated `rag.reranker.model_name` to `ms-marco-MultiBERT-L-12` (multilingual support).
    -   Increased `rag.reranker.top_k` to 15.
    -   Expanded `semantic_keywords` for better dynamic weighting.
    -   Strengthened `analysis_protocol` prompt to enforce strict context usage.
2.  **`src/common/utils.py`**:
    -   Added logging for highlight search queries to aid debugging.
3.  **Untracked Scripts (to be added)**:
    -   `scripts/test_embedding_v2.py`: Verification script for embedding model.
    -   `scripts/verify_reranker_with_pdf.py`: Verification script for reranker using PDF.
    -   `scripts/test_highlight_query_cleaning.py`: (Assuming this was created in a previous step or context, though not explicitly shown in recent turns, it's in untracked files. I will include it as it seems relevant to the "highlight query cleaning" mentioned in the user's prompt about "above evidence sentences".)

## Commit Strategy
I will create a single commit that encapsulates the "Korean RAG Optimization" effort.

**Commit Message:**
```
feat(rag): optimize pipeline for Korean language support

- Replace embedding model with `nomic-embed-text-v2-moe` for better multilingual performance
- Switch reranker to `ms-marco-MultiBERT-L-12` to correctly handle Korean queries
- Increase retriever `top_k` to 15 to reduce context loss
- Expand semantic keywords for dynamic weighting
- Enforce stricter context adherence in system prompts
- Add verification scripts for embedding and reranker
- Add logging for highlight search debugging
```

## Steps
1.  **Stage Changes:** `git add config.yml src/common/utils.py scripts/test_embedding_v2.py scripts/verify_reranker_with_pdf.py scripts/test_highlight_query_cleaning.py`
2.  **Commit:** `git commit -m "..."`
3.  **Push:** `git push origin refactor/ci-test-coverage` (Assuming this is the current branch based on `git status`)

## Verification
- Run `git status` after commit to ensure working directory is clean.
- Run `git log -n 1` to verify the commit.
