import sys
import os
from pathlib import Path

out_dir = Path("graphify-out")
out_dir.mkdir(exist_ok=True)

python_file = out_dir / ".graphify_python"
if not python_file.exists():
    with open(python_file, "w", encoding="utf-8") as f:
        f.write(sys.executable)

print(f"Interpreter resolved: {sys.executable}")
