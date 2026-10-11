# 🔧 Troubleshooting Guide

This guide covers common issues encountered when running the RAG System, particularly on Windows environments.

## 🪟 Windows Specific Issues

### ⚡ Quick Fix
If you encounter crashes or DLL errors, use the Windows-optimized requirements file
(`scripts/fix_windows_env.ps1` was removed — no such script in this repo):

### 1. Application Crashes on Startup (0xc0000139 / DLL Load Failed)

**Symptoms:**
- The application crashes silently or with exit code `0xc0000139` when starting.
- `ImportError: DLL load failed` when importing `torch` or `sentence_transformers`.

**Solutions:**

#### Option A: Install VC++ Redistributable
Download and install the latest supported Visual C++ redistributable packages.
- [Download Link (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### Option B: Clean Install using Windows Requirements
If Option A doesn't work, use our optimized requirements file which excludes problematic packages like `torchvision`:

```bash
pip uninstall torch torchvision
pip install -r requirements-win.txt
```

#### Option C: Run a Quick Smoke Test
No `scripts/diagnose_windows.py` exists in this repo. Check the environment directly:
```bash
python -c "import torch, sentence_transformers; print(torch.__version__)"
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
    ollama pull qwen3:4b-instruct-2507-q4_K_M  # must match models.default_ollama in config.yml
    ```
3.  If running in Docker, ensure `OLLAMA_BASE_URL` is set correctly (usually `http://host.docker.internal:11434` or `http://ollama:11434`).

---

## 💾 Memory & Performance

### 1. CUDA Out of Memory (OOM)
**Error:** `torch.cuda.OutOfMemoryError`

**Solution:**
The system now includes an **Auto-Batch Optimizer**.
1.  Restart the application. The optimizer reads `models.embedding_batch_size` in `config.yml`
    (default `"auto"`) and automatically detects available VRAM to adjust the batch size.
2.  If the issue persists, set an explicit number instead of `"auto"`:
    ```yaml
    models:
      embedding_batch_size: 16  # Try 16 or 8
    ```

### 2. Slow Response Times
- Enable caching in `config.yml`.
- Ensure you are using a GPU if available.
- Check `Performance Monitor` logs for bottlenecks.

---

## ⚡ Concurrency & System Stability

### 1. System Hangs or API Requests Time Out
**Cause:** Deadlock in session management or lock contention during heavy I/O (e.g., indexing large PDFs).

**Solution:**
The system now uses a **Standardized Locking Order** and **Granular Model Locking**.
- If the API still returns a session-lock timeout, check if a background process is stuck. You can manually delete a session via API: `DELETE /api/v1/session/{session_id}`.

### 2. UI Logs/Status Not Updating During Analysis
**Cause:** Streamlit UI thread and background analysis thread using different state stores (State Fragmentation).

**Solution:**
- We have unified the state store into `_fallback_sessions`. The UI now correctly observes background task logs.
- Refresh the browser page if the UI feels stuck; the backend analysis will continue in the background and resume UI logging upon reconnection.

---

## 🐛 Still Stuck?

If these steps don't resolve your issue, please report it with the following info:
1.  Output of `python -c "import torch, sentence_transformers; print(torch.__version__)"`
2.  `logs/app.log` file content
3.  Your `config.yml` settings
