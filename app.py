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

st.title("📄 RAG System with Ollama LLM")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    if os.path.exists("temp.pdf"):
        st.write("✅ PDF 파일이 성공적으로 저장되었습니다.")
    else:
        st.write("❌ PDF 파일 저장에 실패했습니다.")
        st.stop()

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # 로드된 문서 수와 첫 페이지 내용 출력
    st.write(f"📄 로드된 문서(페이지) 수: {len(docs)}")
    if docs:  # 문서가 로드되었다면
        st.write("📄 PDF에서 문서를 추출했습니다.")
    else:
        st.write("❌ PDF에서 문서를 추출하지 못했습니다.")

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
    st.write("🔗 문서 결합 체인이 생성되었습니다.")

    qa = create_retrieval_chain(retriever, combine_documents_chain)
    st.write("🔗 검색 체인이 생성되었습니다.")

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        response = qa.invoke({"input": user_input, "question": user_input})
        
        st.write("**Response:**")
        st.write(response["answer"])