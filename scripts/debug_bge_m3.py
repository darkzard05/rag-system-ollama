import ollama

def test_bge_m3():
    try:
        print("Testing bge-m3 embedding...")
        res = ollama.embed(model="bge-m3", input="This is a simple test sentence.")
        print("Success!")
        print(f"Embedding shape: {len(res['embeddings'][0])}")
        
        print("\nTesting with a slightly longer text...")
        res = ollama.embed(model="bge-m3", input="The bge-m3 model is a versatile embedding model that supports multiple languages and long contexts.")
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bge_m3()