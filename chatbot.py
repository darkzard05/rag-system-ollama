import os
import torch
# Streamlit과 Torch 간의 충돌 해결 패치
torch.classes.__path__ = []

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.title("📄 RAG Chatbot with Ollama LLM")

# PDF 업로드 및 처리
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write("✅ PDF 파일이 성공적으로 저장되었습니다.")

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    st.write(f"📄 로드된 문서(페이지) 수: {len(docs)}")
    if not docs:
        st.error("❌ PDF에서 문서를 추출하지 못했습니다.")
        st.stop()

    embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(docs)
    st.write(f"📑 분할된 문서 청크 수: {len(documents)}")

    vector = FAISS.from_documents(documents, embedder)
    st.write("🗄️ FAISS 벡터 저장소가 생성되었습니다.")

    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.write("🔍 검색기가 생성되었습니다.")

    llm = OllamaLLM(model="deepseek-r1:14b")
    st.write("🤖 LLM이 초기화되었습니다.")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
    QA_PROMPT = PromptTemplate.from_template(prompt)

    combine_documents_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    qa = create_retrieval_chain(retriever, combine_documents_chain)

    # 대화 이력을 저장 (최초 접속 시 기본 인사 메시지 추가)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! PDF 문서 관련 질문을 해주세요."}]

    # 저장된 대화 메시지를 챗 인터페이스로 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자의 입력을 받아 챗봇 대화 진행
    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        # 사용자 메시지 저장 및 출력
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # RAG 체인을 이용해 답변 생성
        response = qa.invoke({"input": user_input, "question": user_input})
        answer = response["answer"]

        # 어시스턴트 메시지 저장 및 출력
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
