# RAG Chatbot with Ollama LLM

![RAG Chatbot Preview](image/image3.png)

## 소개
이 프로젝트는 Ollama LLM을 활용하여 PDF 문서를 처리하고, 문서 기반 질문 응답(RAG, Retrieval-Augmented Generation)을 제공하는 챗봇 애플리케이션입니다. Streamlit을 사용하여 웹 인터페이스를 제공하며, 사용자는 PDF 파일을 업로드하고 해당 문서에 대한 질문을 할 수 있습니다.

## 주요 기능
- **PDF 업로드 및 처리**: 사용자가 업로드한 PDF 파일을 분석하고, 문서 내용을 기반으로 질문에 답변합니다.
- **Ollama LLM 통합**: Ollama LLM을 사용하여 자연어 처리 및 질문 응답을 수행합니다.
- **Streamlit 기반 UI**: 직관적인 사용자 인터페이스를 통해 PDF 미리보기 및 질문 응답 기능을 제공합니다.

## 설치 및 실행

### 요구 사항
- Python 3.10 이상
- Windows 운영 체제

### 설치 방법
1. 이 저장소를 클론합니다:
   ```bash
   git clone github.com/darkzard05/rag-system-ollama.git
   cd rag-system-ollama
   ```

2. 필요한 Python 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

3. Ollama를 설치하고 실행합니다:
   - [Ollama 공식 웹사이트](https://ollama.com)에서 설치 파일을 다운로드하여 설치하세요.
   - 설치 후, `ollama list` 명령어를 실행하여 사용 가능한 모델을 확인합니다.

4. 추천 모델 다운로드:
   ```bash
   ollama pull qwen3:4b
   ```
   - `qwen3:4b` 모델은 이 애플리케이션에서 권장되는 모델입니다.

5. Streamlit 애플리케이션을 실행합니다:
   ```bash
   streamlit run src/main.py
   ```

6. 웹 브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용합니다.

## 파일 구조
```
readme.md
requirements.txt
image/
    image1.png
    image2.png
    image3.png
src/
    main.py
    utils.py
```
- **readme.md**: 프로젝트에 대한 설명 파일입니다.
- **requirements.txt**: 필요한 Python 패키지 목록입니다.
- **image/**: 프로젝트에서 사용하는 이미지 파일이 저장된 폴더입니다.
- **src/**: 애플리케이션의 주요 소스 코드가 포함된 폴더입니다.
  - **main.py**: Streamlit 애플리케이션의 진입점입니다.
  - **utils.py**: PDF 처리 및 기타 유틸리티 함수가 포함된 파일입니다.

## 사용 방법
1. 사이드바에서 PDF 파일을 업로드합니다.
2. PDF 파일이 처리된 후, 문서 내용을 기반으로 질문을 입력합니다.
3. 챗봇이 질문에 대한 답변을 제공합니다.

## 기여
기여를 환영합니다! 버그를 발견하거나 새로운 기능을 제안하려면 이 저장소의 이슈 트래커를 사용하세요.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

