# 📄 RAG Chatbot with Ollama LLM

A **LangChain-based RAG chatbot** that processes PDF documents and answers questions using Ollama LLM.

![RAG Chatbot Screenshot](/image/image2.png)
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
git clone https://github.com/darkzard05/rag-system-ollama.git
cd rag-system-ollama
streamlit run src/main.py
```

## 📑 Features
- **Dynamic Ollama Model Selection**: Choose from available Ollama models via the sidebar.
- **PDF Upload and Processing**: Upload PDF files for text extraction and question answering.
- **PDF Preview with Adjustable Settings**: View uploaded PDFs directly in the app with customizable resolution, width, and height.
- **HuggingFace Embeddings**: Uses `BAAI/bge-m3` or configurable models for embedding generation.
- **FAISS Vector Database**: Efficient document retrieval using FAISS.
- **Customizable QA Prompt**: Modify the QA prompt template for tailored responses.
- **Example Question Generation**: Automatically generates example questions based on the uploaded PDF content.
- **Korean Language Support**: All responses and example questions are generated in Korean.

## 🛠️ Advanced Configuration
- **Cache Management**: Automatic cache invalidation when new PDFs are uploaded or models are changed.
- **Device Configuration**: Default device is set to `cuda` for GPU acceleration. Modify in `utils.py` if needed.
- **Threaded Document Processing**: Parallelized document splitting and vector store creation for faster performance.

## 📝 License
This project is licensed under the MIT License.

