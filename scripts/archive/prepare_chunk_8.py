import json
from pathlib import Path

files = [
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-2.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-3.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-4.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-5.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-6.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-7.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-8.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-7-9.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-8-0.png',
    'logs/pymupdf_test/images/2201.07520v1.pdf-8-1.png'
]

output_path = Path('graphify-out/.graphify_chunk_8_files.txt')
output_path.write_text('\n'.join(files), encoding='utf-8')
print(f'Chunk 8: {len(files)} files written to {output_path}')
