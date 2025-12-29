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
-   **Semantic Chunking:** Supports advanced semantic chunking based on embedding similarity for better context preservation.

## Setup and Installation

### 1. Prerequisites: Install Ollama

You must have Ollama installed and running on your system.

1.  Download and install Ollama from the [official website](https://ollama.com/).
2.  Pull the LLM model you intend to use. The default model in this project is `gemma3:8b` (configurable).
    ```sh
    ollama pull gemma3:8b
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
    ├── semantic_chunker.py# Custom semantic chunking logic
    ├── session.py         # Manages the Streamlit session state
    ├── ui.py              # Contains all Streamlit UI rendering functions
    └── utils.py           # Utility functions (e.g., decorators)
```

## Configuration

The main behavior of the application can be configured in the `config.yml` file:

-   **`models`**: Set the default Ollama model, preferred embedding models, etc.
-   **`rag`**: Configure the RAG pipeline, such as chunk size, retriever settings, semantic chunking parameters, and ensemble weights.
-   **`ui`**: Change UI messages and container heights.

## License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

---
<a name="korean"></a>
# LangGraph & Ollama 기반 RAG 챗봇 (한국어)

**LangGraph, 앙상블 리트리버(BM25 & FAISS), Ollama, Streamlit 기반의 고도화된 PDF 챗봇**

이 프로젝트는 정교한 RAG(Retrieval-Augmented Generation) 시스템을 구현하며, 핵심 파이프라인이 **LangGraph**를 사용한 상태 머신으로 구조화되어 있습니다. 사용자는 PDF 문서를 업로드하고 Ollama를 통해 로컬에서 실행되는 LLM과 대화할 수 있습니다.

![RAG Chatbot Preview](image/image1.png)

## 🔑 주요 기능

- **LangGraph 기반 아키텍처**: RAG 파이프라인(검색 -> 컨텍스트 구성 -> 답변 생성)을 그래프로 구축하여 워크플로우를 투명하고 수정하기 쉽게 만듭니다.
- **고도화된 하이브리드 검색**: 키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)을 결합한 **앙상블 리트리버**를 사용하여 더 정확하고 문맥에 맞는 결과를 제공합니다.
- **의미론적 청킹(Semantic Chunking)**: 단순 길이 기반이 아닌, 임베딩 유사도를 기반으로 문맥 단위로 텍스트를 분할하여 검색 품질을 높입니다.
- **강력한 로컬 LLM (Ollama)**: 개인 정보 보호를 위해 Ollama로 로컬에서 LLM을 실행합니다.
- **효율적인 캐싱**: 벡터 저장소와 모델을 캐싱하여, 앱을 재실행하거나 동일한 파일을 다시 로드할 때 속도가 매우 빠릅니다.
- **대화형 UI**: 손쉬운 문서 업로드, 모델 선택, 채팅 및 나란히 보는 PDF 뷰어를 위해 Streamlit으로 구축된 사용자 친화적인 인터페이스입니다.

---

## ⚡ 빠른 시작

### 📋 사전 준비 사항
- **Python**: 3.10 이상
- **Ollama**: 로컬 모델 사용 시, Ollama가 설치되어 있고 서버가 실행 중이어야 합니다.
  - 설치는 [Ollama 공식 웹사이트](https://ollama.com)를 참조하세요.
- **시스템 리소스**: 로컬 모델(7B 이상) 구동을 위한 충분한 RAM/VRAM이 권장됩니다.

---

### 💻 설치 및 실행

1.  **저장소 클론**
    ```bash
    git clone <repository-url>
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

4.  **Ollama 모델 다운로드**
    기본 설정 모델인 `gemma3:8b`를 다운로드하거나, 원하는 모델을 선택하세요.
    ```bash
    ollama pull gemma3:8b
    ```
    - 이 명령을 실행하기 전에 Ollama 서버가 실행 중인지 확인하세요.
    - `config.yml`의 `default_ollama` 설정을 원하는 모델로 변경할 수 있습니다.

5.  **Streamlit 애플리케이션 실행**
    ```bash
    streamlit run src/main.py
    ```

6.  웹 브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용합니다.

## 📁 파일 구조
```
.
├── config.yml        # 애플리케이션 설정 파일
├── requirements.txt  # 의존성 패키지 목록
└── src/
    ├── main.py            # 앱 진입점
    ├── ui.py              # UI 렌더링 로직
    ├── session.py         # 세션 상태 관리
    ├── rag_core.py        # RAG 핵심 로직 (문서 처리, 검색)
    ├── graph_builder.py   # LangGraph 워크플로우 정의
    ├── semantic_chunker.py# 의미론적 청킹 구현체
    ├── model_loader.py    # 모델 로딩 및 캐싱
    ├── schemas.py         # 데이터 스키마
    ├── config.py          # 설정 로더
    └── utils.py           # 유틸리티 함수
```

## ⚙️ 설정

- **모델 및 파라미터**: `config.yml` 파일에서 다음 항목들을 조정할 수 있습니다.
  - `models`: 기본 LLM 및 임베딩 모델
  - `rag`: 청킹 전략(Semantic Chunking 등), 리트리버 가중치, 검색 파라미터(k)
  - `ui`: UI 메시지 및 레이아웃 설정

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.