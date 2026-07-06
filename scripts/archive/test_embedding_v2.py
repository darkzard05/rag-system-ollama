import sys
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_embedding_performance():
    models_to_test = ["nomic-embed-text", "nomic-embed-text-v2-moe"]
    
    query = "CM3 모델의 주요 특징은 무엇인가요?"
    
    documents = {
        "Relevant (Kor)": "CM3는 텍스트와 이미지 데이터를 동시에 학습하여 멀티모달 생성이 가능한 모델입니다.",
        "Relevant (Eng)": "CM3 is a causally masked generative model trained on structured multi-modal documents.",
        "Irrelevant": "사과의 주성분은 탄수화물이며 비타민 C가 풍부합니다."
    }
    
    print(f"Query: {query}\n")
    
    for model_name in models_to_test:
        print(f"--- Testing Model: {model_name} ---")
        try:
            embedder = OllamaEmbeddings(model=model_name)
            
            # Embed Query
            query_embedding = embedder.embed_query(query)
            
            # Embed Documents
            for label, text in documents.items():
                doc_embedding = embedder.embed_query(text)
                similarity = cosine_similarity(query_embedding, doc_embedding)
                print(f"Similarity with '{label}': {similarity:.4f}")
                
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
        print()

if __name__ == "__main__":
    test_embedding_performance()
