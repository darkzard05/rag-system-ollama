
import pymupdf4llm
import json
import os

def check_pdf_coordinates(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    print(f"Checking coordinates for: {pdf_path}")
    try:
        # extract_words=True ensures word-level coordinates are extracted
        chunks = pymupdf4llm.to_markdown(
            pdf_path, 
            page_chunks=True, 
            extract_words=True
        )
        
        for i, chunk in enumerate(chunks):
            words = chunk.get("words", [])
            print(f"Page {i+1}: Found {len(words)} words with coordinates.")
            if words:
                # Sample first 3 words
                print(f"Sample words for Page {i+1}:")
                for w in words[:3]:
                    print(f"  {w}")
            
            # Check if text is present
            text = chunk.get("text", "")
            print(f"Text length: {len(text)} chars")
            print("-" * 20)
            
            if i >= 1: # Only check first 2 pages for brevity
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use a sample PDF from the project if available
    sample_pdf = "data/temp/sample.pdf" # This might not exist, let's look for one
    
    # Try to find any pdf in the workspace
    import glob
    pdfs = glob.glob("**/*.pdf", recursive=True)
    if pdfs:
        check_pdf_coordinates(pdfs[0])
    else:
        print("No PDF files found in the workspace to test.")
