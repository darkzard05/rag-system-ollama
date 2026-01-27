# 📝 Output Structure Documentation Plan

**Objective:** Upgrade the QA prompt to produce a professional, document-style report output suitable for direct export or formal presentation.

## 1. Proposed Document Structure

The LLM will be instructed to generate a response following this strict Markdown schema:

### Header Section (Metadata)
*   **Title:** H1 Header (`# Title`)
*   **Info Block:** A Markdown table containing:
    *   **Date:** Current date (or "Today")
    *   **Source:** "RAG System Analysis"
    *   **Topic:** Brief keywords from the question

### Body Sections
1.  **Executive Summary (`## 1. Executive Summary`)**
    *   **Head-First:** State the direct answer immediately.
    *   **Overview:** Provide a high-level summary of the context.
2.  **Key Findings (`## 2. Detailed Analysis`)**
    *   **Structured Subsections:** Use `### 2.1`, `### 2.2` for logical grouping.
    *   **Elaboration:** Deep-dive explanations with citations `[p.X]`.
3.  **Conclusion (`## 3. Conclusion`)**
    *   Final synthesis of the information.
4.  **References (`## 4. Source Citations`)**
    *   List of pages used.

## 2. Configuration Changes (`config.yml`)

We will replace the current `qa_system_prompt` with a new version that enforces this schema.

### New Prompt Draft
```yaml
You are a "Senior Technical Writer" tasked with generating a formal documentation report.

[Format Requirement]
Start the response with a metadata table:
| Date | Type | Topic |
| :--- | :--- | :--- |
| Today | Analysis Report | {topic} |

Follow this exact structure:
# [Document Title]

## 1. Executive Summary
...

## 2. Detailed Analysis
### 2.1 [Sub-topic]
...

## 3. Conclusion
...

## 4. References
...
```

## 3. Expected Benefits
*   **Professionalism:** The output looks like a finished product, not just a chat message.
*   **Readability:** Tables and subsections make complex information easier to digest.
*   **Completeness:** Forcing a "Conclusion" section ensures the answer doesn't trail off.

## 4. Action Items
1.  Backup current `config.yml`.
2.  Apply the new prompt structure.
3.  Test with a complex query to verify table formatting and subsections.
