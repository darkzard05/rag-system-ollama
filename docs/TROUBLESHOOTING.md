# 🔧 Troubleshooting Guide

This guide covers common issues encountered when running the RAG System, particularly on Windows environments.

## 🪟 Windows Specific Issues

### 1. Application Crashes on Startup (0xc0000139 / DLL Load Failed)

**Symptoms:**
- The application crashes silently or with exit code `0xc0000139` when starting.
- `ImportError: DLL load failed` when importing `torch` or `sentence_transformers`.

**Cause:**
This is often caused by:
1.  Missing **Microsoft Visual C++ Redistributable** packages.
2.  Incompatible `torchvision` version installed alongside `torch`.
3.  Corrupted PyTorch installation.

**Solutions:**

#### Option A: Install VC++ Redistributable (Recommended)
Download and install the latest supported Visual C++ redistributable packages for Visual Studio 2015, 2017, 2019, and 2022.
- [Download Link (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### Option B: Clean Install without Torchvision
This project handles text and PDF documents, so `torchvision` is generally not required. If you have it installed and are facing crashes:

```bash
pip uninstall torchvision
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
# Or for CUDA support (adjust version as needed)
# pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

#### Option C: Run Diagnosis Script
We provide a script to check your environment:
```bash
python scripts/diagnose_windows.py
```
If `torchvision` fails but `sentence_transformers` succeeds, you can likely ignore the `torchvision` error or uninstall it.

---

## 🤖 Ollama & Model Issues

### 1. "Failed to connect to Ollama"
**Error:** `ConnectionRefusedError` or `httpx.ConnectError`

**Solution:**
1.  Ensure Ollama is running:
    ```bash
    ollama serve
    ```
2.  Check if the model is pulled:
    ```bash
    ollama list
    # If empty or missing target model:
    ollama pull qwen2.5:0.5b  # or your configured model
    ```
3.  If running in Docker, ensure `OLLAMA_BASE_URL` is set correctly (usually `http://host.docker.internal:11434` or `http://ollama:11434`).

---

## 💾 Memory & Performance

### 1. CUDA Out of Memory (OOM)
**Error:** `torch.cuda.OutOfMemoryError`

**Solution:**
The system now includes an **Auto-Batch Optimizer**.
1.  Restart the application. The optimizer will automatically detect available VRAM and adjust `EMBEDDING_BATCH_SIZE` accordingly.
2.  If the issue persists, manually lower the batch size in `config.yml`:
    ```yaml
    embedding:
      batch_size: 16  # Try 16 or 8
    ```

### 2. Slow Response Times
- Enable caching in `config.yml`.
- Ensure you are using a GPU if available.
- Check `Performance Monitor` logs for bottlenecks.

---

## 🐛 Still Stuck?

If these steps don't resolve your issue, please report it with the following info:
1.  Output of `python scripts/diagnose_windows.py`
2.  `logs/app.log` file content
3.  Your `config.yml` settings
