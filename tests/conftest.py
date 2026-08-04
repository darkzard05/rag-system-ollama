"""
Shared pytest fixtures for RAG System tests.
"""

import sys
import uuid
from pathlib import Path

import pytest

# Add src to sys.path
BASE_DIR = Path(__file__).parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mock_llm():
    """Returns a mock LLM for testing."""
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.fixture
def session_context():
    return f"test_session_{uuid.uuid4().hex[:8]}"
