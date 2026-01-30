"""
Shared pytest fixtures for RAG System tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to sys.path
BASE_DIR = Path(__file__).parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def test_pdf_path():
    """Returns the path to the standard test PDF."""
    path = BASE_DIR / "tests" / "data" / "2201.07520v1.pdf"
    return str(path)


@pytest.fixture
def mock_llm():
    """Returns a mock LLM for testing."""
    from unittest.mock import MagicMock

    return MagicMock()
