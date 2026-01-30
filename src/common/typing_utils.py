"""
Custom typing utilities and type aliases for the RAG system.

This module provides:
- TypeVar definitions for generic functions
- Custom Protocol classes for duck typing
- Type aliases for common types
- Generic container types
"""

from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    overload,
)

from langchain_core.documents import Document

# ============================================================================
# TypeVars for Generic Functions
# ============================================================================

# Generic type variables
T = TypeVar("T")  # Generic type
K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type
U = TypeVar("U")  # Another generic type
T_co = TypeVar("T_co", covariant=True)  # Covariant type
T_contra = TypeVar("T_contra", contravariant=True)  # Contravariant type

# Domain-specific type variables
DocumentT = TypeVar("DocumentT", bound=Document)
StateT = TypeVar("StateT")
ConfigT = TypeVar("ConfigT")

# ============================================================================
# Type Aliases
# ============================================================================

# Document and retrieval types
DocumentList = list[Document]
DocumentDict = dict[str, Any]
DocumentDictList = list[DocumentDict]
QueryList = list[str]
ScoreList = list[float]

# Graph and state types
GraphState = dict[str, Any]
GraphOutput = dict[str, Any]
NodeFunction = Callable[[GraphState], GraphOutput]
AsyncNodeFunction = Callable[[GraphState], Any]  # Returns Awaitable

# Configuration and settings
ConfigDict = dict[str, Any]
ModelConfig = dict[str, Any]
EmbeddingConfig = dict[str, Any]
ChunkingConfig = dict[str, Any]

# Batch and performance types
BatchData = list[Union[str, dict[str, Any]]]
BatchResult = list[list[float]]
MemoryInfo = tuple[bool, int]  # (is_available, memory_mb)

# Session types
SessionData = dict[str, Any]
SessionKey = str
SessionValue = Any

# LLM and embedding types
TokenCount = int
EmbeddingVector = list[float]
Embeddings = list[EmbeddingVector]
LogitScores = list[float]

# ============================================================================
# Protocol Definitions (Duck Typing)
# ============================================================================


class Retrievable(Protocol[T_co]):
    """Protocol for objects that can be retrieved."""

    def retrieve(self, query: str, top_k: int) -> list[T_co]:
        """Retrieve items based on a query."""
        ...


class Rankable(Protocol):
    """Protocol for objects that can rank documents."""

    def rank(
        self, query: str, documents: DocumentList
    ) -> tuple[DocumentList, ScoreList]:
        """Rank documents for a given query."""
        ...


class Embeddable(Protocol):
    """Protocol for objects that can generate embeddings."""

    def embed(self, text: str) -> EmbeddingVector:
        """Generate embedding for text."""
        ...

    def embed_batch(self, texts: list[str]) -> Embeddings:
        """Generate embeddings for multiple texts."""
        ...


class GenerativeModel(Protocol):
    """Protocol for generative language models."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on prompt."""
        ...

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate text based on prompt."""
        ...


class Configurable(Protocol):
    """Protocol for configurable objects."""

    def configure(self, config: ConfigDict) -> None:
        """Configure the object with a configuration dictionary."""
        ...

    def get_config(self) -> ConfigDict:
        """Get current configuration."""
        ...


# ============================================================================
# Generic Container Classes
# ============================================================================


class Pipeline(Generic[T, U]):
    """Generic pipeline that transforms input of type T to output of type U."""

    def process(self, input_data: T) -> U:
        """Process input and return output."""
        raise NotImplementedError

    async def aprocess(self, input_data: T) -> U:
        """Asynchronously process input and return output."""
        raise NotImplementedError


class Cache(Generic[K, V]):
    """Generic cache for key-value pairs."""

    def get(self, key: K) -> V | None:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: K, value: V) -> None:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: K) -> None:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError


class Result(Generic[T]):
    """Generic result type for operations that can succeed or fail."""

    def __init__(self, success: bool, value: T | None = None, error: str | None = None):
        self.success = success
        self.value = value
        self.error = error

    def is_ok(self) -> bool:
        """Check if operation succeeded."""
        return self.success

    def is_err(self) -> bool:
        """Check if operation failed."""
        return not self.success

    @staticmethod
    def ok(value: T) -> "Result[T]":
        """Create successful result."""
        return Result(success=True, value=value)

    @staticmethod
    def err(error: str) -> "Result[T]":
        """Create error result."""
        return Result(success=False, error=error)


# ============================================================================
# Overload Definitions
# ============================================================================


@overload
def serialize_value(value: str) -> str: ...


@overload
def serialize_value(value: int) -> str: ...


@overload
def serialize_value(value: float) -> str: ...


@overload
def serialize_value(value: list[T]) -> list[str]: ...


@overload
def serialize_value(value: dict[K, V]) -> dict[K, str]: ...


def serialize_value(value: Any) -> Any:
    """Serialize a value to its string representation.

    Supports multiple input types with proper type hints.
    """
    if isinstance(value, (str, int, float)):
        return str(value)
    elif isinstance(value, list):
        return [str(v) for v in value]
    elif isinstance(value, dict):
        return {k: str(v) for k, v in value.items()}
    else:
        return str(value)


@overload
def get_or_default(data: dict[K, V], key: K) -> V | None: ...


@overload
def get_or_default(data: dict[K, V], key: K, default: U) -> Union[V, U]: ...


def get_or_default(data: dict[K, V], key: K, default: Any = None) -> Any:
    """Get value from dictionary with optional default.

    Returns the value if key exists, otherwise returns default.
    """
    return data.get(key, default)


# ============================================================================
# Type Checking Utilities
# ============================================================================


def is_document(obj: Any) -> bool:
    """Check if object is a Document."""
    return isinstance(obj, Document)


def is_document_list(obj: Any) -> bool:
    """Check if object is a list of Documents."""
    return isinstance(obj, list) and all(isinstance(doc, Document) for doc in obj)


def validate_type(value: T, expected_type: type) -> T:
    """Validate that value is of expected type.

    Raises TypeError if validation fails.
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


# ============================================================================
# TypedDict Definitions (Structured Dictionaries)
# ============================================================================


class RetrievalState(GraphState):
    """State dict for retrieval operations."""

    question: str
    documents: DocumentList
    scores: ScoreList


class GenerationState(GraphState):
    """State dict for generation operations."""

    question: str
    context: str
    answer: str


# ============================================================================
# Union Types for Common Patterns
# ============================================================================

# LLM response types
LLMResponse = Union[str, dict[str, Any]]

# Batch types
Batch = Union[list[str], list[dict[str, Any]]]

# Config types
Config = Union[dict[str, Any], ConfigDict]

# Model types
Model = Union["GenerativeModel", "Embeddable", "Rankable"]

# Error types
ErrorInfo = Union[str, Exception]
