import pytest
from src.common.utils import extract_annotations_from_docs

def test_fuzzy_highlight_matching():
    # Content has 'responses' (plural) while PDF has 'RESPONSE' (singular)
    # Content has 'very' which is missing in PDF
    doc = {
        "page_content": "This is a very test responses",
        "metadata": {
            "page": 1,
            "word_coords": [
                (0, 0, 10, 10, "This"), 
                (11, 0, 20, 10, "is"), 
                (21, 0, 30, 10, "a"), 
                (31, 0, 50, 10, "TEST"), 
                (51, 0, 100, 10, "RESPONSE.") 
            ],
            "file_path": "dummy.pdf"
        }
    }
    
    annos = extract_annotations_from_docs([doc])
    
    assert len(annos) > 0
    assert annos[0]["page"] == 1
    print(f"\nFound {len(annos)} annotations")
