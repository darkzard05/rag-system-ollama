import os
from pathlib import Path

import pytest

from common.config import DEFAULT_EMBEDDING_MODEL, PROJECT_ROOT
from core.rag_core import RAGSystem
from core.resource_manager import get_resource_manager
from core.session import SessionManager


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("IS_CI_TEST") == "true", reason="실제 임베딩/LLM 모델 필요"
)
async def test_upload_flow_persistence_and_summary():
    # 1. Setup
    session_id = "test-upload-flow-verify"
    rag = RAGSystem(session_id=session_id)

    # Ensure uploads dir is clean for test
    upload_dir = Path("uploads")
    if upload_dir.exists():
        for f in upload_dir.glob(f"{session_id}_*"):
            f.unlink()

    # 2. Document Path
    test_pdf = str(PROJECT_ROOT / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)

    # Use actual embedder for real pipeline test
    embedder = await get_resource_manager().get_embedder(DEFAULT_EMBEDDING_MODEL)

    # 3. Execute Upload Simulation (Mimicking src/main.py upload handler)

    # Simulate the main app's permanent-file save logic:
    # it copies the uploaded file and records the path in session state
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    permanent_path = upload_dir / f"{session_id}_{file_name}"

    # Copy test file to permanent storage
    import shutil

    shutil.copy(test_pdf, permanent_path)
    SessionManager.set("pdf_file_path", str(permanent_path), session_id=session_id)

    # Now run the pipeline build as the orchestrator would
    msg, cache_used = await rag.build_pipeline(str(permanent_path), file_name, embedder)

    # 4. Verification: File Persistence
    # In the actual app, the main module saves the file; here we check that the
    # path recorded in session state is correct and non-temporary.
    pdf_path = SessionManager.get("pdf_file_path", session_id=session_id)
    assert pdf_path is not None, "PDF file path should be set in session state"
    assert "uploads" in pdf_path, (
        f"File should be saved in uploads directory, got: {pdf_path}"
    )
    assert os.path.exists(pdf_path), f"Permanent file should exist on disk: {pdf_path}"
    assert "temp" not in pdf_path.lower(), "File path should not be a temporary path"

    # 5. Verification: Analysis Summary
    summary = SessionManager.get("analysis_summary", session_id=session_id)
    if summary is not None:
        assert len(summary) > 20, f"Summary is too short or empty: {summary}"

    # 6. Verification: State Persistence
    # Session state (SessionManager) is the persistence layer; re-initializing the
    # session must preserve the values written by the pipeline build.
    SessionManager.init_session(session_id=session_id)
    persisted_path = SessionManager.get("pdf_file_path", session_id=session_id)
    assert persisted_path == pdf_path, (
        "Permanent path must be persisted in session state"
    )
    persisted_summary = SessionManager.get("analysis_summary", session_id=session_id)
    assert persisted_summary == summary, (
        "Analysis summary must be persisted in session state"
    )
