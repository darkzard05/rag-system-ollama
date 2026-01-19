# Local RAG System with Ollama & LangGraph

This project is a sophisticated **Retrieval-Augmented Generation (RAG)** application that enables you to chat with your PDF documents using local Large Language Models (LLMs) via **Ollama**. Built with **Streamlit** for the frontend and **LangGraph** for the backend logic, it ensures data privacy, high performance, and an interactive user experience.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Key Features

-   **Local & Private:** Runs entirely on your machine using [Ollama](https://ollama.com/), keeping your documents secure.
-   **LangGraph Architecture:** The RAG pipeline is modeled as a state machine (graph), making the workflow of retrieval, context generation, and answering transparent and extensible.
-   **Hybrid Search (Ensemble Retriever):** Combines **BM25** (keyword-based) and **FAISS** (semantic vector-based) search to retrieve the most relevant context.
-   **Semantic Chunking:** Implements advanced chunking strategies based on embedding similarity, ensuring that text is split by meaning rather than arbitrary length.
-   **Configurable Embeddings:** Supports various Sentence-Transformer models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-large-instruct`).
-   **Advanced RAG Options:**
    -   **Reranking:** Optional step to re-score retrieved documents using a Cross-Encoder for higher precision (configurable in `config.yml`).
    -   **Query Expansion:** Optional multi-query generation to broaden search scope (configurable in `config.yml`).
-   **Interactive UI:** Feature-rich interface with a built-in PDF viewer alongside the chat window, supporting citation tooltips.
-   **Efficient Caching:** Caches vector stores and models to minimize loading times for previously processed files.

> **Note:** The default configuration (`config.yml`) and UI system prompts are currently set to **Korean**. You can modify the `config.yml` file to change prompts and messages to English if needed.

## 📋 Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: You must have Ollama installed and running.
    -   Download from [ollama.com](https://ollama.com/).
    -   Pull the default model (or any model you prefer):
        ```bash
        ollama pull qwen3:4b
        ```

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd rag-system-ollama
    ```

2.  **Create a Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Start Ollama Server**
    Ensure Ollama is running in the background.

2.  **Run the Application**
    ```bash
    streamlit run src/main.py
    ```

3.  **Interact**
    -   Open your browser at `http://localhost:8501`.
    -   **Sidebar:** Upload a PDF and select your preferred LLM/Embedding models.
    -   **Chat:** Once processed, ask questions about the document.

## ⚙️ Configuration

The application is highly configurable via `config.yml`.

| Section | Key | Description |
| :--- | :--- | :--- |
| **models** | `default_ollama` | Default LLM model to use (e.g., `qwen3:4b`). |
| | `available_embeddings` | List of supported embedding models. |
| | `ollama_num_predict` | Max output tokens (-1 for default/infinite). |
| **rag** | `ensemble_weights` | Weights for [BM25, FAISS] (e.g., `[0.4, 0.6]`). |
| | `semantic_chunker` | Enable/disable semantic chunking and set thresholds. |
| | `reranker` | Enable/disable re-ranking step (requires more VRAM). |
| | `query_expansion` | Enable/disable generating multiple queries. |
| **ui** | `container_height` | Adjust the height of the chat/PDF window. |

## 📂 Project Structure

```
.
├── config.yml           # Central configuration file (Default settings are in Korean)
├── requirements.txt     # Python dependencies
├── src/
│   ├── main.py          # Streamlit app entry point
│   ├── rag_core.py      # Core RAG logic (loading, splitting, embedding)
│   ├── graph_builder.py # LangGraph workflow definition
│   ├── semantic_chunker.py # Custom semantic chunking implementation
│   ├── model_loader.py  # Model management and caching
│   ├── session.py       # Streamlit session state management
│   ├── ui.py            # UI components and layout
│   ├── schemas.py       # Data types and state definitions
│   └── utils.py         # Helper functions
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
