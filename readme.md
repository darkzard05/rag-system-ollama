# RAG System with Ollama & LangGraph

This project is a Streamlit web application that allows you to chat with your PDF documents using a local Large Language Model (LLM) powered by Ollama. It leverages the RAG (Retrieval-Augmented Generation) pattern, LangChain, and LangGraph to provide contextual answers based on the content of your uploaded files.

## Key Features

-   **Chat with Your PDFs:** Upload a PDF file and ask questions about its content.
-   **Powered by Local LLMs:** Uses Ollama to run LLMs locally on your machine, ensuring privacy and control.
-   **Selectable Embedding Models:** Choose from a list of different sentence-transformer models for document embedding.
-   **Hybrid Search:** Combines dense vector search (FAISS) and keyword-based search (BM25) for robust and accurate document retrieval.
-   **Built with LangGraph:** The RAG pipeline is orchestrated as a graph, making the logic clear and extensible.
-   **Interactive UI:** A user-friendly interface built with Streamlit, including a PDF viewer.
-   **Caching:** Caches processed documents (vector stores) to ensure fast re-loading of previously analyzed files.

## Setup and Installation

### 1. Prerequisites: Install Ollama

You must have Ollama installed and running on your system.

1.  Download and install Ollama from the [official website](https://ollama.com/).
2.  Pull the LLM model you intend to use. The default model in this project is `qwen2:1.5b`.
    ```sh
    ollama pull qwen2:1.5b
    ```

### 2. Clone the Repository

```sh
git clone <repository-url>
cd rag-system-ollama
```

### 3. Create a Virtual Environment (Recommended)

```sh
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from `requirements.txt`.

```sh
pip install -r requirements.txt
```

## How to Run

Once the setup is complete, you can run the Streamlit application with the following command:

```sh
streamlit run src/main.py
```

The application will open in your default web browser.

## Project Structure

```
.
├── .env.example
├── config.yml
├── readme.md
├── requirements.txt
└── src/
    ├── config.py          # Loads settings from config.yml
    ├── graph_builder.py   # Defines the RAG workflow using LangGraph
    ├── main.py            # Main entry point for the Streamlit app
    ├── model_loader.py    # Handles loading LLMs and embedding models
    ├── rag_core.py        # Core RAG logic (document loading, splitting, embedding)
    ├── schemas.py         # Data schemas (e.g., GraphState)
    ├── session.py         # Manages the Streamlit session state
    ├── ui.py              # Contains all Streamlit UI rendering functions
    └── utils.py           # Utility functions (e.g., decorators)
```

## Configuration

The main behavior of the application can be configured in the `config.yml` file:

-   **`models`**: Set the default Ollama model, preferred embedding models, etc.
-   **`rag`**: Configure the RAG pipeline, such as chunk size, retriever settings, and ensemble weights.
-   **`ui`**: Change UI messages and container heights.


---
<a name="english"></a>
# RAG Chatbot with LangGraph, Ollama & Gemini (English)

**An advanced PDF-based Chatbot powered by LangGraph, an Ensemble Retriever (BM25 & FAISS), Ollama, Gemini, and Streamlit.**

This project implements a sophisticated RAG (Retrieval-Augmented Generation) system where the core pipeline is structured as a state machine using **LangGraph**. This approach enhances modularity, observability, and scalability.

![RAG Chatbot Preview](image/image1.png)

## 🔑 Key Features

- **LangGraph-based Architecture**: The RAG pipeline (Retrieve -> Format Context -> Generate Response) is built as a graph, making the workflow transparent and easy to modify.
- **Workflow Visualization**: The Streamlit UI includes a "Workflow" tab that visually represents the RAG pipeline structure using Mermaid diagrams, offering clear insight into the operational flow.
- **Advanced Hybrid Search**: Utilizes an **Ensemble Retriever** that combines keyword-based search (BM25) and semantic search (FAISS) to deliver more accurate and contextually relevant results.
- **Flexible LLM Selection (Ollama & Gemini)**: Choose between running large language models locally with Ollama for privacy, or using powerful Gemini models via API for high performance.
- **Efficient Caching**: The system caches the entire FAISS vector store, not just document splits. This significantly speeds up initialization by avoiding the need to re-calculate embeddings for previously processed documents.
- **View LLM's Thinking Process**: An expander in the UI shows the internal "thought" process of the LLM before it generates a final answer, providing transparency into its reasoning.
- **Interactive UI**: A user-friendly interface built with Streamlit for easy document upload, model selection, chatting, and side-by-side PDF viewing.

---

## ⚡ Quick Start

### 📋 Prerequisites
- **Python**: 3.10 or higher
- **Ollama (Optional)**: If using local models, Ollama must be installed and the server running.
  - Refer to the [Ollama Official Website](https://ollama.com) for installation.
- **Gemini API Key (Optional)**: If using Gemini models, you need to set up your API key.
  - Create a `.env` file in the project's root directory.
  - Add your API key to the file: `GEMINI_API_KEY="YOUR_API_KEY"`
  - You can get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- **System Resources**: For local models, sufficient RAM (e.g., 16GB+ for 7B models) is recommended.

---

### 💻 Installation & Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/darkzard05/rag-system-ollama.git
    cd rag-system-ollama
    ```

2.  **(Recommended) Create and activate a virtual environment**
    ```bash
    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install required Python packages**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(For Ollama Users) Pull a model**
    We recommend a general-purpose model to start. You can find more models at the [Ollama Library](https://ollama.com/library).
    ```bash
    ollama pull llama3:8b
    ```
    - Ensure the Ollama server is running before this step.
    - Make sure the `default_ollama` model name in your `config.yml` matches a model you have pulled.

5.  **Run the Streamlit application**
    ```bash
    streamlit run src/main.py
    ```

6.  Open your web browser and go to `http://localhost:8501` to use the application.

## 📁 Project Structure
```
.
├── ...
├── requirements.txt
├── image/
│   └── image1.png
└── src/
    ├── main.py
    ├── ui.py
    ├── session.py
    ├── rag_core.py
    ├── graph_builder.py  <-- New: Defines the LangGraph structure
    ├── schemas.py        <-- New: Defines data structures (e.g., GraphState)
    ├── config.py
    └── config.yml
```
- **`main.py`**: Entry point of the Streamlit application.
- **`ui.py`**: Renders the Streamlit UI, including the chat interface and workflow visualization.
- **`session.py`**: Manages the application's session state.
- **`rag_core.py`**: Handles core RAG logic like document processing, embedding, and retriever creation.
- **`graph_builder.py`**: Constructs the RAG workflow using LangGraph.
- **`schemas.py`**: Defines the state object used throughout the graph.
- **`config.py` & `config.yml`**: Manage application configurations.

## ✨ Key Components

- **LangGraph**: Orchestrates the RAG pipeline as a stateful graph.
- **PDF Loader (PyMuPDF)**: Loads and extracts text from PDF files.
- **Text Splitter (Langchain)**: Divides text into smaller, manageable chunks.
- **Embedding Model (Sentence Transformers)**: Converts text chunks into vector embeddings.
- **Vector Store (FAISS)**: Stores embeddings for efficient similarity search.
- **Retriever (Ensemble)**: Combines **BM25** (keyword) and **FAISS** (semantic) search.
- **LLM (Ollama & Gemini)**: Generates answers based on user queries and retrieved context.
- **Streamlit UI**: Provides the interactive web interface.

## ⚙️ Configuration

- **API Keys**: Set your `GEMINI_API_KEY` in a `.env` file.
- **Models and Parameters**: Adjust models, retriever weights, text splitter settings, and **Ollama's response token limit (`ollama_num_predict`)** in `config.yml`. The default `ollama_num_predict` is set to `2048` to balance detailed responses with performance.

## 📄 License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

---
<a name="korean"></a>
# LangGraph, Ollama & Gemini 기반 RAG 챗봇 (한국어)

**LangGraph, 앙상블 리트리버(BM25 & FAISS), Ollama, Gemini, Streamlit 기반의 고도화된 PDF 챗봇**

이 프로젝트는 정교한 RAG(Retrieval-Augmented Generation) 시스템을 구현하며, 핵심 파이프라인이 **LangGraph**를 사용한 상태 머신으로 구조화되어 있습니다. 이 접근 방식은 모듈성, 관찰 가능성 및 확장성을 향상시킵니다.

![RAG Chatbot Preview](image/image1.png)

## 🔑 주요 기능

- **LangGraph 기반 아키텍처**: RAG 파이프라인(검색 -> 컨텍스트 구성 -> 답변 생성)을 그래프로 구축하여 워크플로우를 투명하고 수정하기 쉽게 만듭니다.
- **워크플로우 시각화**: Streamlit UI에 RAG 파이프라인 구조를 Mermaid 다이어그램으로 시각적으로 표현하는 "워크플로우" 탭을 포함하여, 작동 흐름에 대한 명확한 통찰력을 제공합니다.
- **고도화된 하이브리드 검색**: 키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)을 결합한 **앙상블 리트리버**를 사용하여 더 정확하고 문맥에 맞는 결과를 제공합니다.
- **유연한 LLM 선택 (Ollama & Gemini)**: 개인 정보 보호를 위해 Ollama로 로컬에서 LLM을 실행하거나, 고성능을 위해 API를 통해 강력한 Gemini 모델을 사용하는 것 중에서 선택할 수 있습니다.
- **효율적인 캐싱**: 시스템은 문서 조각뿐만 아니라 전체 FAISS 벡터 저장소를 캐시합니다. 이를 통해 이전에 처리된 문서에 대한 임베딩을 다시 계산할 필요가 없어 초기화 속도가 크게 향상됩니다.
- **LLM의 사고 과정 확인**: UI의 확장 패널을 통해 LLM이 최종 답변을 생성하기 전의 내부 "사고" 과정을 보여주어 추론 과정의 투명성을 제공합니다.
- **대화형 UI**: 손쉬운 문서 업로드, 모델 선택, 채팅 및 나란히 보는 PDF 뷰어를 위해 Streamlit으로 구축된 사용자 친화적인 인터페이스입니다.

---

## ⚡ 빠른 시작

### 📋 사전 준비 사항
- **Python**: 3.10 이상
- **Ollama (선택 사항)**: 로컬 모델 사용 시, Ollama가 설치되어 있고 서버가 실행 중이어야 합니다.
  - 설치는 [Ollama 공식 웹사이트](https://ollama.com)를 참조하세요.
- **Gemini API 키 (선택 사항)**: Gemini 모델 사용 시, API 키를 설정해야 합니다.
  - 프로젝트 루트 디렉터리에 `.env` 파일을 생성하세요.
  - 파일에 `GEMINI_API_KEY="YOUR_API_KEY"` 형식으로 API 키를 추가하세요.
  - [Google AI Studio](https://aistudio.google.com/app/apikey)에서 키를 발급받을 수 있습니다.
- **시스템 리소스**: 로컬 모델의 경우, 충분한 RAM(예: 7B 모델의 경우 16GB 이상)이 권장됩니다.

---

### 💻 설치 및 실행

1.  **저장소 클론**
    ```bash
    git clone https://github.com/darkzard05/rag-system-ollama.git
    cd rag-system-ollama
    ```

2.  **(권장) 가상 환경 생성 및 활성화**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필요한 Python 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Ollama 사용자) 모델 다운로드**
    시작을 위해 범용 모델을 사용하는 것을 권장합니다. 더 많은 모델은 [Ollama 라이브러리](https://ollama.com/library)에서 찾을 수 있습니다.
    ```bash
    ollama pull llama3:8b
    ```
    - 이 명령을 실행하기 전에 Ollama 서버가 실행 중인지 확인하세요.
    - `config.yml`의 `default_ollama` 모델 이름이 다운로드한 모델과 일치하는지 확인하세요.

5.  **Streamlit 애플리케이션 실행**
    ```bash
    streamlit run src/main.py
    ```

6.  웹 브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용합니다.

## 📁 파일 구조
```
.
├── ...
├── requirements.txt
├── image/
│   └── image1.png
└── src/
    ├── main.py
    ├── ui.py
    ├── session.py
    ├── rag_core.py
    ├── graph_builder.py  <-- 신규: LangGraph 구조 정의
    ├── schemas.py        <-- 신규: 데이터 구조(예: GraphState) 정의
    ├── config.py
    └── config.yml
```
- **`main.py`**: Streamlit 애플리케이션의 진입점입니다.
- **`ui.py`**: 채팅 인터페이스 및 워크플로우 시각화를 포함한 Streamlit UI를 렌더링합니다.
- **`session.py`**: 애플리케이션의 세션 상태를 관리합니다.
- **`rag_core.py`**: 문서 처리, 임베딩, 리트리버 생성과 같은 핵심 RAG 로직을 처리합니다.
- **`graph_builder.py`**: LangGraph를 사용하여 RAG 워크플로우를 구성합니다.
- **`schemas.py`**: 그래프 전체에서 사용되는 상태 객체를 정의합니다.
- **`config.py` & `config.yml`**: 애플리케이션 설정을 관리합니다.

## ✨ 주요 구성 요소

- **LangGraph**: RAG 파이프라인을 상태 기반 그래프로 조율합니다.
- **PDF 로더 (PyMuPDF)**: PDF 파일에서 텍스트를 로드하고 추출합니다.
- **텍스트 분할기 (Langchain)**: 텍스트를 관리하기 쉬운 작은 청크로 나눕니다.
- **임베딩 모델 (Sentence Transformers)**: 텍스트 청크를 벡터 임베딩으로 변환합니다.
- **벡터 저장소 (FAISS)**: 효율적인 유사도 검색을 위해 임베딩을 저장합니다.
- **리트리버 (Ensemble)**: **BM25**(키워드)와 **FAISS**(의미) 검색을 결합합니다.
- **LLM (Ollama & Gemini)**: 사용자 쿼리와 검색된 컨텍스트를 기반으로 답변을 생성합니다.
- **Streamlit UI**: 대화형 웹 인터페이스를 제공합니다.

## ⚙️ 설정

- **API 키**: `.env` 파일에 `GEMINI_API_KEY`를 설정하세요.
- **모델 및 파라미터**: `config.yml` 파일에서 모델, 리트리버 가중치, 텍스트 분할기 설정 등을 조정할 수 있습니다.

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.