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
streamlit run chatbot.py
```
Access at `http://localhost:8501`.

## 📑 Features
- PDF upload and text extraction
- HuggingFace embeddings (`intfloat/e5-base-v2`)
- FAISS vector database for search
- Dynamic Ollama model selection via sidebar

## 📝 License
This project is licensed under the MIT License.

