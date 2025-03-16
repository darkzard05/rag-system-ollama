# 📄 RAG Chatbot with Ollama LLM

This project is a **LangChain-based RAG (Retrieval-Augmented Generation) chatbot**.  
Upload a PDF document to convert its content into embedding vectors and get answers to questions using Ollama LLM.

---

## 🚀 1️⃣ Installation

### 📌 **1. Environment Setup**  
- Requires **Python 3.9 or higher**.  
- Ollama installation needed (for local LLM execution).

### 📌 **2. Install Ollama and Download Model**  
#### 🔹 **Install Ollama** (check the latest version)  
👉 Download from [Ollama Official Site](https://ollama.com) or run in terminal:  
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### 🔹 **Download LLM Model**  
After installing Ollama, download the desired model.  
By default, the `gemma3:4b` model is used.

```bash
ollama pull gemma3:4b
```
---

## 📌 2️⃣ Install Packages  
Run in the project root directory:

```bash
pip install -r requirements.txt
```

---

## 🎯 3️⃣ How to Run  
```bash
streamlit run chatbot.py
```
After running, access `http://localhost:8501` in your browser.

---

## 📑 4️⃣ Key Features  
- **PDF document upload and text extraction**  
- **HuggingFace Embedding Model** (`intfloat/e5-base-v2`)  
- **FAISS vector database** for document search  
- **Ollama LLM** for Q&A  

---

## 🔧 5️⃣ How to Change Settings  
### ✅ **Using a Different LLM Model**  
By default, `gemma3:4b` is used, but you can change it in the `config.py` file.  
Modify the following in `config.py`:

```python
LLM_MODEL = "deepseek-r1:14b"  # gemma3:4b --> deepseek-r1:14b
```

💡 **Check Available Models**  
```bash
ollama list
```

---

## 📝 6️⃣ License  
This project is licensed under the MIT License.

