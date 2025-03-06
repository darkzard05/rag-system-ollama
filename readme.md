# 📄 RAG Chatbot with Ollama LLM

이 프로젝트는 **LangChain 기반 RAG (Retrieval-Augmented Generation) 챗봇**입니다.  
PDF 문서를 업로드하면 내용을 임베딩 벡터로 변환하고, Ollama LLM을 활용해 질문에 대한 답변을 제공합니다.  

---

## 🚀 1️⃣ 설치 방법  

### 📌 **1. 환경 설정**  
- **Python 3.9 이상**이 필요합니다.  
- Ollama 설치 필요 (로컬 LLM 실행용)  

### 📌 **2. Ollama 설치 및 모델 다운로드**  
#### 🔹 **Ollama 설치** (최신 버전 확인 후 설치)  
👉 [Ollama 공식 사이트](https://ollama.com) 에서 다운로드하거나, 터미널에서 실행:  
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### 🔹 **LLM 모델 다운로드**  
Ollama를 설치한 후, 원하는 모델을 다운로드하세요.  
기본적으로 `deepseek-r1:14b` 모델을 사용합니다.

```bash
ollama pull deepseek-r1:14b
```
🚀 **다른 모델을 사용하고 싶다면?**  
예를 들어 `mistral` 모델을 사용하려면:  
```bash
ollama pull mistral
```

---

## 📌 2️⃣ 패키지 설치  
프로젝트 루트 디렉토리에서 실행하세요.  

```bash
pip install -r requirements.txt
```

---

## 🎯 3️⃣ 실행 방법  
```bash
streamlit run chatbot.py
```
실행 후 브라우저에서 `http://localhost:8501` 접속

---

## 📑 4️⃣ 주요 기능  
- **PDF 문서 업로드 및 텍스트 추출**  
- **HuggingFace Embedding 모델** (`intfloat/e5-base-v2`) 사용  
- **FAISS 벡터 데이터베이스** 활용한 문서 검색  
- **Ollama LLM (`deepseek-r1:14b`)을 이용한 질의응답**  

---

## 🔧 5️⃣ 설정 변경 방법  
### ✅ **다른 LLM 모델 사용하기**  
기본적으로 `deepseek-r1:14b`을 사용하지만, 다른 모델로 변경할 수 있습니다.  
`app.py`에서 아래 부분을 수정하세요.  

```python
llm = OllamaLLM(model="mistral")  # deepseek-r1:14b → mistral
```

💡 **사용 가능한 모델 목록 확인**  
```bash
ollama list
```

---

## 📂 6️⃣ 폴더 구조  
```
/rag-chatbot-ollama  
│── app.py  # Streamlit 앱 메인 코드  
│── requirements.txt  # 필요한 패키지 목록  
│── README.md  # 프로젝트 설명  
│── .gitignore  # 불필요한 파일 제외 (예: temp.pdf)  
```

---

## 📝 7️⃣ 라이선스  
이 프로젝트는 MIT 라이선스를 따릅니다.

