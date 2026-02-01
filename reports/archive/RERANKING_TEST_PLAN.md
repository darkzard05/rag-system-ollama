# 🧪 Reranking Performance Test Plan

**Objective:** Determine the optimal balance between response quality and generation speed by varying `top_k` (retrieval count) and `rerank_top_k` (final context count).

## 1. Test Variables

We will test the following combinations of parameters:

| Experiment ID | Retriever `top_k` | Reranker `top_k` | Reranker Enabled | Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 5 | N/A | ❌ Off | Fast, but lower accuracy. |
| **Exp-A** | 10 | 3 | ✅ On | High precision, low LLM latency. |
| **Exp-B** | 10 | 5 | ✅ On | Balanced approach. |
| **Exp-C** | 20 | 5 | ✅ On | Better recall, slightly slower reranking. |
| **Exp-D** | 30 | 7 | ✅ On | Max context, slowest speed. |

## 2. Metrics

### 2.1 Latency Metrics (Time-based)
*   **Retrieval Time:** Time to fetch documents from Vector DB + BM25.
*   **Reranking Time:** Time taken by the Cross-Encoder model.
*   **Total Turnaround Time:** End-to-end time from user query to final answer.

### 2.2 Quality Proxies
*   **Context Relevance Score:** Average score of the documents selected by the reranker.
*   **Answer Length:** Token count of the generated answer (longer answers often imply more detailed context usage).
*   **Citation Count:** Number of distinct pages cited in the answer (e.g., `[p.1]`, `[p.5]`).

## 3. Test Dataset (Sample Questions)
Select 5 representative questions from your target PDF document:
1.  **Fact Retrieval:** "What is the specific value of X?"
2.  **Summarization:** "Summarize the key points of section Y."
3.  **Comparison:** "Compare the pros and cons of A and B."
4.  **Reasoning:** "Why did the author suggest Z?"
5.  **Detail Extraction:** "List all the requirements mentioned in Chapter 3."

## 4. Execution Strategy

### 4.1 Automated Script (`tests/benchmark_reranking.py`)
Create a Python script that:
1.  Loads the RAG system.
2.  Iterates through the parameter combinations.
3.  Runs the 5 questions for each combination.
4.  Logs the start/end times and result metrics.
5.  Saves the report to `reports/RERANKING_BENCHMARK.csv`.

### 4.2 Analysis
After running the script, we will analyze the CSV to find the "Sweet Spot" where quality gains plateau and latency spikes.

## 5. Next Steps
1.  Create the benchmark script.
2.  Execute the benchmark.
3.  Analyze results and update `config.yml` with the winner.
