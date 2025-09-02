[Read in English](#english) | [한국어로 보기](#korean)

---
<a name="english"></a>
# RAG Chatbot with Ollama & Gemini (English)

**An advanced PDF-based Chatbot powered by an Ensemble Retriever (BM25 & FAISS), Ollama, Gemini, and Streamlit.**

![RAG Chatbot Preview](image/image1.png)

## 🔑 Key Features

- **PDF-based Q&A**: Upload your PDF documents and get answers to your questions based on their content.
- **Advanced Hybrid Search**: Utilizes an **Ensemble Retriever** that combines keyword-based search (BM25) and semantic search (FAISS) to deliver more accurate and contextually relevant results.
- **Flexible LLM Selection (Ollama & Gemini)**: Choose between running large language models locally with Ollama for privacy, or using powerful Gemini models via API for high performance.
- **Flexible Embedding Model Selection**: Choose from a variety of open-source embedding models to suit your needs for performance and language support.
- **Streamlit-based Web Interface**: A user-friendly and interactive web interface for easy document upload, model selection, chatting, and PDF viewing.
- **View LLM's Thinking Process**: Option to see the thought process of the LLM before it generates an answer, providing transparency into its reasoning.

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
├── .env.example
├── .gitignore
├── LICENSE
├── readme.md
├── requirements.txt
├── image/
│   └── image1.png
└── src/
    ├── main.py
    ├── ui.py
    ├── session.py
    ├── rag_core.py
    ├── config.py
    └── config.yml
```
- **`main.py`**: Entry point of the Streamlit application.
- **`ui.py`**: Contains all functions for rendering the Streamlit user interface.
- **`session.py`**: Manages the application's session state.
- **`rag_core.py`**: The core of the RAG system (data processing, embedding, retrieval, QA chain).
- **`config.py`**: Loads and provides configuration constants for the application.
- **`config.yml`**: YAML file for storing configurations like model lists and retriever settings.
- **`.env.example`**: An example file for environment variables. Copy it to `.env` to set your API keys.

## ✨ Key Components

- **PDF Loader (PyMuPDF)**: Loads and extracts text content from uploaded PDF files.
- **Text Splitter (Langchain)**: Divides the extracted text into smaller, manageable chunks.
- **Embedding Model (Sentence Transformers)**: Converts text chunks into numerical vector embeddings.
- **Vector Store (FAISS)**: Stores vector embeddings for efficient similarity search.
- **Retriever (Ensemble)**: Combines keyword-based search (**BM25**) and semantic search (**FAISS**) to fetch the most relevant text chunks for a given query.
- **LLM (Ollama & Gemini)**: The selected language model generates an answer using the user's query and the retrieved context.
- **Streamlit UI**: Provides the interactive web interface for all user interactions.

## ⚙️ Configuration

- **API Keys**: Set your `GEMINI_API_KEY` in a `.env` file in the project root.
- **Models and Parameters**: You can adjust the models, retriever weights, and text splitter settings in `config.yml`.

## 🚑 Troubleshooting

- **Ollama Connection Issues**: Ensure the Ollama application/server is running (`ollama list`).
- **Gemini API Key Issues**: Ensure the key is correct and set in your `.env` file. A `429` error might indicate you have exceeded your API rate limits.
- **Slow Performance**: Processing large PDFs or using large local models can be resource-intensive. Ensure your system meets the recommended specifications.

## 📄 License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

---
<a name="korean"></a>
# RAG Chatbot with Ollama & Gemini (한국어)

**앙상블 리트리버(BM25 & FAISS), Ollama, Gemini, Streamlit 기반의 고도화된 PDF 챗봇**

![RAG Chatbot Preview](image/image1.png)

## 🔑 주요 기능

- **PDF 기반 Q&A**: PDF 문서를 업로드하고 해당 내용을 기반으로 질문에 대한 답변을 얻을 수 있습니다.
- **고도화된 하이브리드 검색**: 키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)을 결합한 **앙상블 리트리버**를 사용하여 더 정확하고 문맥에 맞는 결과를 제공합니다.
- **유연한 LLM 선택 (Ollama & Gemini)**: 개인 정보 보호를 위해 Ollama로 로컬에서 LLM을 실행하거나, 고성능을 위해 API를 통해 강력한 Gemini 모델을 사용하는 것 중에서 선택할 수 있습니다.
- **유연한 임베딩 모델 선택**: 성능 및 언어 지원 요구에 맞는 다양한 오픈소스 임베딩 모델 중에서 선택할 수 있습니다.
- **Streamlit 기반 웹 인터페이스**: 손쉬운 문서 업로드, 모델 선택, 채팅, PDF 뷰잉을 위한 사용자 친화적인 대화형 웹 인터페이스입니다.
- **LLM의 사고 과정 확인**: LLM이 답변을 생성하기 전의 사고 과정을 확인할 수 있는 옵션을 제공하여 추론 과정의 투명성을 높입니다.

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
├── .env.example
├── .gitignore
├── LICENSE
├── readme.md
├── requirements.txt
├── image/
│   └── image1.png
└── src/
    ├── main.py
    ├── ui.py
    ├── session.py
    ├── rag_core.py
    ├── config.py
    └── config.yml
```
- **`main.py`**: Streamlit 애플리케이션의 진입점입니다.
- **`ui.py`**: Streamlit 사용자 인터페이스 렌더링 함수들을 포함합니다.
- **`session.py`**: 애플리케이션의 세션 상태를 관리합니다.
- **`rag_core.py`**: RAG 시스템의 핵심 로직(데이터 처리, 임베딩, 리트리버, QA 체인)을 담당합니다.
- **`config.py`**: 애플리케이션의 설정 상수를 로드하고 제공합니다.
- **`config.yml`**: 모델 목록, 리트리버 설정 등 주요 설정을 저장하는 YAML 파일입니다.
- **`.env.example`**: 환경 변수 예시 파일입니다. 이 파일을 `.env`로 복사하여 API 키 등을 설정할 수 있습니다.

## ✨ 주요 구성 요소

- **PDF 로더 (PyMuPDF)**: 업로드된 PDF 파일에서 텍스트 내용을 로드하고 추출합니다.
- **텍스트 분할기 (Langchain)**: 추출된 텍스트를 처리하기 쉬운 작은 청크로 나눕니다.
- **임베딩 모델 (Sentence Transformers)**: 텍스트 청크를 숫자 벡터 임베딩으로 변환합니다.
- **벡터 저장소 (FAISS)**: 벡터 임베딩을 저장하여 효율적인 유사도 검색을 수행합니다.
- **리트리버 (Ensemble)**: 키워드 기반 검색(**BM25**)과 의미 기반 검색(**FAISS**)을 결합하여 주어진 쿼리에 가장 관련성 높은 텍스트 청크를 가져옵니다.
- **LLM (Ollama & Gemini)**: 선택된 언어 모델이 사용자의 쿼리와 검색된 컨텍스트를 사용하여 답변을 생성합니다.
- **Streamlit UI**: 모든 사용자 상호 작용을 위한 대화형 웹 인터페이스를 제공합니다.

## ⚙️ 설정

- **API 키**: 프로젝트 루트에 `.env` 파일을 만들고 `GEMINI_API_KEY`를 설정하세요.
- **모델 및 파라미터**: `config.yml` 파일에서 모델, 리트리버 가중치, 텍스트 분할기 설정 등을 조정할 수 있습니다.

## 🚑 문제 해결

- **Ollama 연결 문제**: Ollama 애플리케이션/서버가 실행 중인지 확인하세요 (`ollama list`).
- **Gemini API 키 문제**: 키가 올바르고 `.env` 파일에 정확히 설정되었는지 확인하세요. `429` 오류는 API 할당량을 초과했음을 의미할 수 있습니다.
- **느린 성능**: 대용량 PDF를 처리하거나 큰 로컬 모델을 사용하는 것은 리소스를 많이 소모할 수 있습니다. 시스템이 권장 사양을 충족하는지 확인하세요.

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
