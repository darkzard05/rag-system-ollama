"""Probe qwen3 thinking/reasoning emission against the LIVE Ollama server.

Todo 2 of ``.omo/plans/chat-expander-empty-body.md`` — machine-readable
evidence of the root cause behind the RAG app's empty ``thought`` field.

Evidence-only script. It MUST NOT import from ``src/`` and MUST NOT modify any
``src/`` file. It is NOT the CI stub path (no ``IS_CI_TEST``) — it talks to the
real Ollama at ``http://127.0.0.1:11434``.

What it does
------------
(a) POST ``/api/chat`` (``stream: true``) with ``qwen3:4b-instruct-2507-q4_K_M``
    TWICE — once with ``"think": true``, once without — using a trivial prompt
    (``2+2=?``). For each chunk it records whether a ``thinking`` / ``reasoning``
    / ``reasoning_content`` key appears on ``message``, plus ``eval_count`` and
    ``total_duration`` from the final chunk.
(b) Prints installed versions of ``langchain_ollama``, ``langchain_core``,
    ``streamlit`` (both ``importlib.metadata`` and the runtime ``__version__``).
(c) Inspects ``ChatOllama`` accepted fields: are ``stream_content_blocks`` and
    ``reasoning`` model fields (``model_fields`` / ``model_json_schema()``)?
(d) Runs ``ollama show <model> --modelfile`` via subprocess (captured, failure
    tolerated) and records the template / PARAMETER lines.

Outputs
-------
- ``.omo/evidence/expander_empty_body/probe_qwen3_thinking.json``
- ``.omo/evidence/expander_empty_body/task-2-chat-expander-empty-body.txt``
  (verbatim console transcript)

Exit codes: 0 on success; 1 if Ollama is unreachable (both probes error) or a
hard runtime failure occurs.
"""

from __future__ import annotations

import importlib.metadata
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
PROMPT = "2+2=?"
STREAM_TIMEOUT_S = 60
MODELFILE_TIMEOUT_S = 30

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".omo" / "evidence" / "expander_empty_body"
JSON_PATH = EVIDENCE_DIR / "probe_qwen3_thinking.json"
TXT_PATH = EVIDENCE_DIR / "task-2-chat-expander-empty-body.txt"

# requirements/base.txt pins for context (lines 2, 12, 14).
PINS = {
    "streamlit": "==1.60.0 (base.txt:2)",
    "langchain-core": ">=0.3.34,<0.4.0 (base.txt:12)",
    "langchain-ollama": ">=0.2.2 (base.txt:14)",
}

_log: list[str] = []


def emit(line: str = "") -> None:
    """Print a line and record it for the console-transcript file."""
    print(line)
    _log.append(line)


def probe_stream(payload: dict) -> dict:
    """Stream one /api/chat request; return the recorded observations."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/api/chat", data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    thinking_key_count = 0
    thinking_keys_seen: set[str] = set()
    chunk_count = 0
    content_parts: list[str] = []
    eval_count: int | None = None
    total_duration_ns: int | None = None
    final_chunk: dict | None = None

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=STREAM_TIMEOUT_S) as resp:  # noqa: S310 - local Ollama
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                emit(f"  [warn] non-JSON stream line: {line[:200]!r}")
                continue
            chunk_count += 1
            msg = chunk.get("message") or {}
            for key in ("thinking", "reasoning", "reasoning_content"):
                if key in msg:
                    thinking_key_count += 1
                    thinking_keys_seen.add(key)
            content = msg.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
            if chunk.get("done"):
                final_chunk = chunk
                eval_count = chunk.get("eval_count")
                total_duration_ns = chunk.get("total_duration")
    elapsed = time.monotonic() - t0

    return {
        "chunk_count": chunk_count,
        "thinking_key_count": thinking_key_count,
        "thinking_keys_seen": sorted(thinking_keys_seen),
        "content_head": "".join(content_parts)[:200],
        "eval_count": eval_count,
        "total_duration_ns": total_duration_ns,
        "total_duration_ms": (
            round(total_duration_ns / 1_000_000, 1) if total_duration_ns else None
        ),
        "final_chunk_keys": sorted(final_chunk.keys()) if final_chunk else [],
        "elapsed_s": round(elapsed, 2),
    }


def run_probe(label: str, payload: dict) -> tuple[dict, Exception | None]:
    """Run one probe, emitting a human-readable trace; never raises on I/O."""
    emit(f"[probe] {label}: POST {BASE_URL}/api/chat stream=true")
    emit(f"        payload keys: {sorted(payload.keys())}")
    try:
        result = probe_stream(payload)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
    ) as exc:
        emit(f"        ERROR: {type(exc).__name__}: {exc}")
        return {"payload": payload, "error": f"{type(exc).__name__}: {exc}"}, exc
    emit(
        f"        chunks={result['chunk_count']} thinking_key_count={result['thinking_key_count']}"
        f" keys={result['thinking_keys_seen']}"
    )
    emit(
        f"        eval_count={result['eval_count']} total_duration_ms={result['total_duration_ms']}"
        f" elapsed_s={result['elapsed_s']}"
    )
    emit(f"        content_head={result['content_head']!r}")
    return result, None


def installed_versions() -> dict:
    """Record installed package versions (metadata + runtime import)."""
    versions: dict = {"by_importlib_metadata": {}, "by_runtime_import": {}}
    for pkg in ("langchain-ollama", "langchain-core", "streamlit"):
        try:
            versions["by_importlib_metadata"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions["by_importlib_metadata"][pkg] = None

    import langchain_core
    import langchain_ollama
    import streamlit

    versions["by_runtime_import"] = {
        "langchain_ollama": getattr(langchain_ollama, "__version__", None),
        "langchain_core": getattr(langchain_core, "__version__", None),
        "streamlit": getattr(streamlit, "__version__", None),
    }
    return versions


def accepted_fields() -> dict:
    """Report whether ChatOllama declares stream_content_blocks / reasoning.

    ``model_fields`` is the primary source. ``model_json_schema()`` can raise
    ``PydanticInvalidForJsonSchema`` on pydantic 2.13 (langchain-core 1.4.9)
    for callable fields, so the schema inspection is best-effort.
    """
    from langchain_ollama import ChatOllama

    field_names = set(ChatOllama.model_fields.keys())
    result: dict = {
        "model_fields_count": len(field_names),
        "stream_content_blocks_in_model_fields": "stream_content_blocks" in field_names,
        "reasoning_in_model_fields": "reasoning" in field_names,
        "schema_inspection": None,
    }
    try:
        schema_props = set(ChatOllama.model_json_schema().get("properties", {}).keys())
        result["schema_inspection"] = {
            "stream_content_blocks_in_schema": "stream_content_blocks" in schema_props,
            "reasoning_in_schema": "reasoning" in schema_props,
        }
    except Exception as exc:  # pragma: no cover - pydantic version drift
        result["schema_inspection"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }
    return result


def modelfile_excerpt() -> dict:
    """Capture `ollama show <model> --modelfile`; tolerate any failure."""
    cmd = ["ollama", "show", MODEL, "--modelfile"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=MODELFILE_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"command": cmd, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {
            "command": cmd,
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}",
        }
    lines = proc.stdout.splitlines()
    return {
        "command": cmd,
        "ok": True,
        "template": "\n".join(ln for ln in lines if ln.startswith("TEMPLATE"))[:2000],
        "parameters": [ln for ln in lines if ln.startswith("PARAMETER")],
        "excerpt": "\n".join(lines[:80])[:4000],
    }


def main() -> int:
    try:
        socket.setdefaulttimeout(STREAM_TIMEOUT_S)

        emit("== probe_qwen3_thinking (T2 chat-expander-empty-body) ==")
        emit(
            f"model={MODEL} base_url={BASE_URL} prompt={PROMPT!r} timeout={STREAM_TIMEOUT_S}s"
        )
        emit(f"timestamp={datetime.now(timezone.utc).isoformat()}")

        # Quick reachability gate for a clear, early error message.
        try:
            with urllib.request.urlopen(
                f"{BASE_URL}/api/tags", timeout=10
            ) as tags_resp:  # noqa: S310
                tags_json = json.load(tags_resp)
            models_present = [m.get("name") for m in tags_json.get("models", [])]
            emit(f"[reachability] /api/tags OK — models: {models_present}")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
        ) as exc:
            emit(f"[reachability] /api/tags FAILED: {type(exc).__name__}: {exc}")
            emit(
                "Hint: is Ollama running? Try `curl.exe -s http://127.0.0.1:11434/api/tags`"
            )
            return 1

        # (a) Two stream probes — think:true and think absent.
        payload_think = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "think": True,
        }
        payload_no_think = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
        }
        result_think, err_think = run_probe("think=true", payload_think)
        emit("")
        result_no_think, err_no_think = run_probe("think=absent", payload_no_think)
        emit("")

        if err_think is not None and err_no_think is not None:
            emit("FATAL: both probes errored — Ollama unreachable. Exiting 1.")
            return 1

        # (b) Installed versions.
        versions = installed_versions()
        emit("[versions]")
        emit(f"  by_importlib_metadata: {versions['by_importlib_metadata']}")
        emit(f"  by_runtime_import:     {versions['by_runtime_import']}")

        # (c) ChatOllama accepted fields.
        fields = accepted_fields()
        emit("[accepted_fields] ChatOllama")
        emit(
            f"  stream_content_blocks in model_fields={fields['stream_content_blocks_in_model_fields']}"
            "  <- assumption under test"
        )
        emit(
            f"  reasoning             in model_fields={fields['reasoning_in_model_fields']}"
        )
        schema_info = fields.get("schema_inspection") or {}
        if "error" in schema_info:
            emit(f"  model_json_schema() failed (recorded): {schema_info['error']}")
        else:
            emit(
                f"  schema: stream_content_blocks={schema_info.get('stream_content_blocks_in_schema')}"
                f" reasoning={schema_info.get('reasoning_in_schema')}"
            )

        # (d) Modelfile.
        modelfile = modelfile_excerpt()
        if modelfile["ok"]:
            emit(f"[modelfile] `ollama show {MODEL} --modelfile` OK")
            for p in modelfile["parameters"]:
                emit(f"  {p}")
            emit(f"  template: {modelfile['template'][:400]!r}")
        else:
            emit(f"[modelfile] FAILED (tolerated): {modelfile['error']}")

        # Assemble + persist the machine-readable evidence.
        report = {
            "probe": {
                "todo": 2,
                "plan": "chat-expander-empty-body",
                "model": MODEL,
                "base_url": BASE_URL,
                "prompt": PROMPT,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stream_timeout_s": STREAM_TIMEOUT_S,
                "notes": (
                    "qwen3:4b-instruct-2507-q4_K_M on Ollama. think:true vs no-think "
                    "probes over /api/chat stream=true."
                ),
            },
            "requests": {
                "think_true": {
                    **result_think,
                    "payload_think_true": True,
                },
                "think_absent": result_no_think,
            },
            "versions": versions,
            "pins": PINS,
            "accepted_fields": fields,
            "modelfile_excerpt": modelfile,
        }

        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        TXT_PATH.write_text("\n".join(_log) + "\n", encoding="utf-8")

        emit("")
        emit(f"[written] {JSON_PATH}")
        emit(f"[written] {TXT_PATH}")

        k_think = result_think.get("thinking_key_count")
        k_absent = result_no_think.get("thinking_key_count")
        emit("")
        emit("== RESULT ==")
        emit(f"think=true  -> thinking_key_count = {k_think}")
        emit(f"think=absent-> thinking_key_count = {k_absent}")
        if k_think == 0 and k_absent == 0:
            emit(
                "EXPECTED confirmed: the model emits NO separated thinking/reasoning "
                "in either probe -> root cause: model does not emit thought, "
                "consistent with the T6 gate baseline."
            )
        else:
            emit(
                "!!! NONZERO thinking_key_count detected -> this CHANGES the T6 gate. "
                "Flag prominently in the todo output."
            )
        emit("exit=0")
        return 0
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        emit(f"FATAL unexpected error: {type(exc).__name__}: {exc}")
        try:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            TXT_PATH.write_text("\n".join(_log) + "\n", encoding="utf-8")
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
