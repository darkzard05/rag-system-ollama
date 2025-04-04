# 📄 RAG Chatbot with Ollama LLM

A **LangChain-based RAG chatbot** that processes PDF documents and answers questions using Ollama LLM.

![RAG Chatbot Screenshot](/image/image1.png)
*A screenshot of the RAG Chatbot application in action. The image shows the user interface, including the sidebar for model selection, the PDF upload area, and the chat interface where questions can be asked and answered.*

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
- **HuggingFace Embeddings**: Uses `BAAI/bge-m3` or configurable models for embedding generation.
- **FAISS Vector Database**: Efficient document retrieval using FAISS.
- **Customizable QA Prompt**: Modify the QA prompt template for tailored responses.
- **Real-Time PDF Preview**: View uploaded PDFs directly in the app with adjustable resolution and dimensions.

## 🛠️ Advanced Configuration
- **Cache Management**: Automatic cache invalidation when new PDFs are uploaded or models are changed.
- **Device Configuration**: Default device is set to `cuda` for GPU acceleration. Modify in `utils.py` if needed.
- **Threaded Document Processing**: Parallelized document splitting and vector store creation for faster performance.

## 📝 License
This project is licensed under the MIT License.

