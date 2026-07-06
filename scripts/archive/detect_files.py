import json
import sys
from pathlib import Path
from graphify.detect import detect

# Use the interpreter we just saved
with open("graphify-out/.graphify_python", "r", encoding="utf-8") as f:
    python_exe = f.read().strip()

# Run detection
result = detect(Path("."))

# Write result
with open("graphify-out/.graphify_detect.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

# Summary
total_files = result.get('total_files', 0)
total_words = result.get('total_words', 0)
files = result.get('files', {})

summary = f"Corpus: {total_files} files · ~{total_words:,} words\n"
for cat, paths in files.items():
    if paths:
        summary += f"  {cat}: {len(paths)} files\n"

print(summary)
print(json.dumps(result)) # Still need the JSON for Step 3 logic
