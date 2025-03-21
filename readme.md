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
- **PDF Upload and Processing**  
  Upload a PDF file, and the chatbot extracts and processes its content for question answering. The processing status is dynamically updated on the main page.
  
- **Dynamic Ollama Model Selection**  
  Select an Ollama LLM model from the sidebar. The chatbot supports dynamic model switching, and the selected model is displayed in the chat history.

- **HuggingFace Embeddings**  
  Uses `intfloat/e5-base-v2` embeddings for semantic chunking and vector representation.

- **FAISS Vector Database**  
  Efficient similarity search using FAISS for document retrieval.

- **Real-Time Chat Interface**  
  All messages, including user inputs and assistant responses, are displayed in chronological order on the main page.

- **Error Handling**  
  Provides clear error messages for issues like PDF processing failures, vector store creation errors, or LLM initialization problems.

## 📝 License
This project is licensed under the MIT License.

