[Read in English](#english) | [한국어로 보기](#korean)

---
<a name="english"></a>
# RAG Chatbot with Ollama & Gemini (English)

**PDF-based Chatbot powered by Ollama, Gemini, and Streamlit**

![RAG Chatbot Preview](image/image1.png)

## 🔑 Key Features

- **PDF-based Q&A**: Upload your PDF documents and get answers to your questions based on their content.
- **Flexible LLM Selection (Ollama & Gemini)**: Choose between running large language models locally with Ollama for privacy, or using the powerful Gemini 1.5 Pro model via API for high performance.
- **Flexible Embedding Model Selection**: Choose from a variety of embedding models to suit your needs for performance and language support.
- **Streamlit-based Web Interface**: A user-friendly and interactive web interface built with Streamlit for easy document upload, chatting, and PDF viewing.
- **View LLM's Thinking Process**: Option to see the thought process of the LLM before it generates an answer, providing transparency.

---

## ⚡ Quick Start

### 📋 Prerequisites
- **Python**: 3.10 or higher
- **Ollama (Optional)**: If using local models, Ollama must be installed and the server running.
  - Refer to the [Ollama Official Website](https://ollama.com) for installation.
- **Gemini API Key (Optional)**: If using the Gemini model, you need to set up your API key.
  - Create a `.env` file in the project's root directory.
  - Add your API key to the file like this: `GEMINI_API_KEY="YOUR_API_KEY"`
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
    
    # On Windows (cmd.exe)
    venv\Scripts\activate
    
    # On Windows (PowerShell)
    # Make sure execution policy is set, e.g., Set-ExecutionPolicy Unrestricted -Scope Process
    venv\Scripts\Activate.ps1
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install required Python packages**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(For Ollama Users) Install and run Ollama**
    - Download and install from the [Ollama Official Website](https://ollama.com).
    - After installation, ensure the Ollama server is running (`ollama list`).

5.  **(For Ollama Users) Download the recommended model**
    ```bash
    ollama pull qwen3:4b
    ```
    - The `qwen3:4b` model is the default for local processing.

6.  **Run the Streamlit application**
    ```bash
    streamlit run src/main.py
    ```

7.  Open your web browser and go to `http://localhost:8501` to use the application.

## 📁 Project Structure
```
.
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
    └── config.py
```
- **`readme.md`**: Project description file.
- **`requirements.txt`**: List of required Python packages.
- **`image/`**: Folder containing project images.
- **`src/`**: Folder containing main application source code.
  - **`main.py`**: Entry point of the Streamlit application. It initializes the app and orchestrates the different modules.
  - **`ui.py`**: Contains all functions related to rendering the Streamlit user interface (sidebar, chat area, PDF viewer).
  - **`session.py`**: Manages the application's session state, including chat history, processing status, and selected models.
  - **`rag_core.py`**: The core of the RAG system. Handles PDF loading, text splitting, embedding, vector store creation, and building the QA chain.
  - **`config.py`**: Contains all configuration constants for the application, such as model names, retriever settings, and text splitter parameters.

## ✨ Key Components

- **PDF Loader (PyMuPDF)**: Loads and extracts text content from uploaded PDF files.
- **Text Splitter (Langchain)**: Divides the extracted text into smaller, manageable chunks for processing.
- **Embedding Model (Sentence Transformers)**: Converts text chunks into numerical vector embeddings. This project uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` by default (supports multiple languages).
- **Vector Store (FAISS)**: Stores these embeddings and allows for efficient similarity searches to find relevant chunks based on a user's query.
- **Retriever (Ensemble)**: Combines keyword-based search (BM25) and semantic search (FAISS) to fetch the most relevant text chunks.
- **LLM (Ollama & Gemini)**: The selected language model generates an answer using the user's query and the retrieved context.
  - **Ollama**: Runs models like `qwen3:4b` locally.
  - **Gemini**: Uses `gemini-1.5-flash` via the `ChatGoogleGenerativeAI` integration.
- **Streamlit UI**: Provides the interactive web interface for all user interactions.

## 📝 How to Use
1.  **Upload a PDF file** in the sidebar.
2.  **Select an LLM model** from the dropdown menu. (Note: For Gemini models, ensure your API key is set in the `.env` file as described in the Prerequisites.)
3.  **Select an embedding model** from the dropdown menu.
4.  Wait for the PDF to be processed. A notification will appear.
5.  Enter your questions about the document content in the chat input field.
6.  The chatbot will provide answers. You can expand the "🤔 Thinking Process" section below each answer to see the LLM's reasoning steps.

## ⚙️ Configuration

- **LLM Model**: Selectable via the UI. Default is `qwen3:4b` (Ollama) and `gemini-1.5-flash` (Gemini).
- **Embedding Model**: Currently set to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in `src/config.py`.

## 🛠️ Technical Stack

- **Programming Language**: Python 3.10+
- **LLM Orchestration**: Langchain, Langchain-Google-GenAI
- **LLM Providers**: Ollama, Google Gemini
- **Web Framework**: Streamlit
- **Embedding Models**: Sentence Transformers (Hugging Face)
- **PDF Processing**: PyMuPDF
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Core ML/DL**: PyTorch

## 🚑 Troubleshooting

- **Ollama Connection Issues**:
  - Ensure the Ollama application/server is running (`ollama list`).
- **Gemini API Key Issues**:
  - Ensure the key is correct, has the necessary permissions, and is set in your `.env` file.
  - A `429` error might indicate you have exceeded your API rate limits.
- **Slow Performance**:
  - Processing large PDFs or using large local models can be resource-intensive.
  - Ensure your system meets the recommended specifications (especially RAM for local models).

## 🤝 Contributing
Contributions are welcome! If you find a bug or want to suggest a new feature, please use the issue tracker in this repository.

## 📄 License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

---
<a name="korean"></a>
# RAG Chatbot with Ollama & Gemini (한국어)

**PDF 기반 챗봇 (Ollama, Gemini, Streamlit)**

![RAG Chatbot Preview](image/image1.png)

## 🔑 주요 기능

- **PDF 기반 Q&A**: PDF 문서를 업로드하고 해당 내용을 기반으로 질문에 대한 답변을 얻을 수 있습니다.
- **유연한 LLM 선택 (Ollama & Gemini)**: 개인 정보 보호를 위해 Ollama로 로컬에서 LLM을 실행하거나, 고성능을 위해 API를 통해 강력한 Gemini 1.5 Pro 모델을 사용하는 것 중에서 선택할 수 있습니다.
- **유연한 임베딩 모델 선택**: 성능 및 언어 지원 요구에 맞는 다양한 임베딩 모델 중에서 선택할 수 있습니다.
- **Streamlit 기반 웹 인터페이스**: Streamlit으로 구축된 사용자 친화적이고 대화형 웹 인터페이스를 통해 손쉽게 문서를 업로드하고, 채팅하며, PDF를 확인할 수 있습니다.
- **LLM의 사고 과정 확인**: LLM이 답변을 생성하기 전의 사고 과정을 확인할 수 있는 옵션을 제공하여 투명성을 높입니다.

---

## ⚡ 빠른 시작

### 📋 사전 준비 사항
- **Python**: 3.10 이상
- **Ollama (선택 사항)**: 로컬 모델 사용 시, Ollama가 설치되어 있고 서버가 실행 중이어야 합니다.
  - 설치는 [Ollama 공식 웹사이트](https://ollama.com)를 참조하세요.
- **Gemini API 키 (선택 사항)**: Gemini 모델 사용 시, API 키를 설정해야 합니다.
  - 프로젝트 루트 디렉터리에 `.env` 파일을 생성하세요.
  - 파일에 다음과 같이 API 키를 추가하세요: `GEMINI_API_KEY="YOUR_API_KEY"`
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
    
    # Windows (cmd.exe)
    venv\Scripts\activate
    
    # Windows (PowerShell)
    # 실행 정책이 설정되어 있는지 확인하세요. 예: Set-ExecutionPolicy Unrestricted -Scope Process
    venv\Scripts\Activate.ps1
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필요한 Python 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Ollama 사용자) Ollama 설치 및 실행**
    - [Ollama 공식 웹사이트](https://ollama.com)에서 설치 후, 서버가 실행 중인지 확인하세요 (`ollama list`).

5.  **(Ollama 사용자) 추천 모델 다운로드**
    ```bash
    ollama pull qwen3:4b
    ```
    - `qwen3:4b` 모델은 로컬 처리를 위한 기본값입니다.

6.  **Streamlit 애플리케이션 실행**
    ```bash
    streamlit run src/main.py
    ```

7.  웹 브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용합니다.

## 📁 파일 구조
```
.
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
    └── config.py
```
- **`readme.md`**: 프로젝트에 대한 설명 파일입니다.
- **`requirements.txt`**: 필요한 Python ���키지 목록입니다.
- **`image/`**: 프로젝트에서 사용하는 이미지 파일이 저장된 폴더입니다.
- **`src/`**: 애플리케이션의 주요 소스 코드가 포함된 폴더입니다.
  - **`main.py`**: Streamlit 애플리케이션의 진입점입니다. 앱을 초기화하고 다른 모듈들을 조립하는 역할을 합니다.
  - **`ui.py`**: 사이드바, 채팅 영역, PDF 뷰어 등 Streamlit 사용자 인터페이스 렌더링과 관련된 모든 함수를 포함합니다.
  - **`session.py`**: 채팅 기록, 처리 상태, 선택된 모델 등 애플리케이션의 세션 상태를 관리합니다.
  - **`rag_core.py`**: RAG 시스템의 핵심입니다. PDF 로딩, 텍스트 분할, 임베딩, 벡터 저장소 생성, QA 체인 구성 등을 담당합니다.
  - **`config.py`**: 모델 이름, 리트리버 설정, 텍스트 분할 파라미터 등 애플리케이션의 모든 설정 상수를 포함합니다.

## ✨ 주요 구성 요소

- **PDF 로더 (PyMuPDF)**: 업로드된 PDF 파일에서 텍스트 내용을 로드하고 추출합니다.
- **텍스트 분할기 (Langchain)**: 추출된 텍스트를 처리하기 쉬운 작은 청크로 나눕니다.
- **임베딩 모델 (Sentence Transformers)**: 텍스트 청크를 숫자 벡터 임베딩으로 변환합니다. 이 프로젝트는 기본적으로 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`를 사용합니다 (다국어 지원).
- **벡터 저장소 (FAISS)**: 이러한 임베딩을 저장하고 사용자 쿼리를 기반으로 관련 청크를 효율적으로 검색할 수 있도록 합니다.
- **리트리버 (Ensemble)**: 키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)을 결합하여 가장 관련성 높은 텍스트 청크를 가져옵니다.
- **LLM (Ollama & Gemini)**: 선택된 언어 모델이 사용자의 쿼리와 검색된 컨텍스트를 사용하여 답변을 생성합니다.
  - **Ollama**: `qwen3:4b`와 같은 모델을 로컬에서 실행합니다.
  - **Gemini**: `ChatGoogleGenerativeAI` 통합을 통해 `gemini-1.5-flash`를 사용합니다.
- **Streamlit UI**: 모든 사용자 상호 작용을 위한 대화형 웹 인터페이스를 제공합니다.

## 📝 사용 방법
1.  사이드바에서 **PDF 파일을 업로드**합니다.
2.  드롭다운 메뉴에서 **LLM 모델을 선택**합니다. (참고: Gemini 모델의 경우, 사전 준비 사항에 설명된 대로 `.env` 파일에 API 키가 설정되어 있는지 확인하세요.)
3.  드롭다운 메뉴에서 **임베딩 모델을 선택**합니다.
4.  PDF가 처리될 때까지 기다립니다. 알림이 표시됩니다.
5.  채팅 입력창에 문서 내용에 대한 질문을 입력합니다.
6.  챗봇��� 답변을 제공합니다. 각 답변 아래의 "🤔 생각 과정" 섹션을 확장하여 LLM의 추론 단계를 확인할 수 있습니다.

## ⚙️ 설정

- **LLM 모델**: UI를 통해 선택 가능. 기본값은 `qwen3:4b`(Ollama) 및 `gemini-1.5-flash`(Gemini)입니다.
- **임베딩 모델**: 현재 `src/config.py`에서 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`로 설정되어 있습니다.

## 🛠️ 기술 스택

- **프로그래밍 언어**: Python 3.10+
- **LLM 오케스트레이션**: Langchain, Langchain-Google-GenAI
- **LLM 제공자**: Ollama, Google Gemini
- **웹 프레임워크**: Streamlit
- **임베딩 모델**: Sentence Transformers (Hugging Face)
- **PDF 처리**: PyMuPDF
- **벡터 저장소**: FAISS (Facebook AI Similarity Search)
- **핵심 ML/DL**: PyTorch

## 🚑 문제 해결

- **Ollama 연결 문제**:
  - Ollama 애플리케이션/서버가 실행 중인지 확인하세요 (`ollama list`).
- **Gemini API 키 문제**:
  - 키가 올바르고, 필요한 권한을 가졌으며, `.env` 파일에 설정되어 있는지 확인하세요.
  - `429` 오류는 API 할당량을 초과했음을 의미할 수 있습니다.
- **느린 성능**:
  - 대용량 PDF를 처리하거나 대규모 로컬 모델을 사용하는 것은 리소스를 많이 소모할 수 있습니다.
  - 시스템이 권장 사양(특히 로컬 모델의 경우 RAM)을 충족하는지 확인하세요.

## 🤝 기여
기여를 환영합니다! 버그를 발견하거나 새로운 기능을 제안하려면 이 저장소의 이슈 트래커를 사용하세요.

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.