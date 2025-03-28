# 📄 RAG Chatbot with Ollama LLM

A **LangChain-based RAG chatbot** that processes PDF documents and answers questions using Ollama LLM.

## 🚀 Installation

### 1. Prerequisites
- **Python 3.9+**
- Install **Ollama** for local LLM execution:  
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download LLM Model
```bash
ollama pull gemma3:4b
```
💡 Check available models:  
```bash
ollama list
```

## 🎯 Usage
Run the chatbot:
```bash
streamlit run src/main.py
```
Access at `http://localhost:8501`.

## 📑 Features
- **Dynamic Ollama Model Selection**: Choose from available Ollama models via the sidebar.
- **PDF Upload and Processing**: Upload PDF files for text extraction and question answering.
- **HuggingFace Embeddings**: Uses `intfloat/e5-base-v2` or configurable models for embedding generation.
- **FAISS Vector Database**: Efficient document retrieval using FAISS.
- **Threaded Document Processing**: Parallelized document splitting and vector store creation for faster performance.
- **Customizable QA Prompt**: Modify the QA prompt template for tailored responses.

## 🛠️ Advanced Configuration
- **Cache Management**: Automatic cache invalidation when new PDFs are uploaded.
- **Device Configuration**: Default device is set to `cuda` for GPU acceleration. Modify in `utils.py` if needed.

## 📝 License
This project is licensed under the MIT License.

