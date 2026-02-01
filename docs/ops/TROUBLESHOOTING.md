# 🔧 Troubleshooting Guide

This guide covers common issues encountered when running the RAG System, particularly on Windows environments.

## 🪟 Windows Specific Issues

### ⚡ Quick Fix (Recommended)
If you encounter crashes or DLL errors, we provide an automatic fix script:
```powershell
# 관리자 권한으로 실행하거나 아래 명령어를 입력하세요.
powershell ./scripts/fix_windows_env.ps1
```
이 스크립트는 충돌하는 패키지를 삭제하고 Windows 안정 버전을 재설치합니다.

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

## ⚡ Concurrency & System Stability

### 1. System Hangs or API Requests Time Out
**Cause:** Deadlock in session management or lock contention during heavy I/O (e.g., indexing large PDFs).

**Solution:**
The system now uses a **Standardized Locking Order** and **Granular Model Locking**.
- Ensure you are on version 2.1.0 or later.
- If the API still returns `SessionLockTimeoutError`, check if a background process is stuck. You can manually delete a session via API: `DELETE /api/v1/session/{id}`.

### 2. UI Logs/Status Not Updating During Analysis
**Cause:** Streamlit UI thread and background analysis thread using different state stores (State Fragmentation).

**Solution:**
- We have unified the state store into `_fallback_sessions`. The UI now correctly observes background task logs.
- Refresh the browser page if the UI feels stuck; the backend analysis will continue in the background and resume UI logging upon reconnection.

---

## 🐛 Still Stuck?

If these steps don't resolve your issue, please report it with the following info:
1.  Output of `python scripts/diagnose_windows.py`
2.  `logs/app.log` file content
3.  Your `config.yml` settings
