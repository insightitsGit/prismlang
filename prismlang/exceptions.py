"""PrismLang exception hierarchy.

All exceptions inherit from ``PrismLangError`` so callers can catch the
entire family with a single ``except PrismLangError`` clause, or catch
specific subtypes for targeted error handling.

Example::

    from prismlang.exceptions import TaxonomyError, EncoderError

    try:
        slug = taxonomy.infer_category(text)
    except TaxonomyError as exc:
        logger.error("Category inference failed: %s", exc)
"""

from __future__ import annotations


class PrismLangError(Exception):
    """Base class for all PrismLang errors."""


# ── Encoder ────────────────────────────────────────────────────────

class EncoderError(PrismLangError):
    """Raised when the ONNX encoder fails to initialise or run inference."""


class ModelDownloadError(EncoderError):
    """Raised when the ONNX model cannot be downloaded from HuggingFace."""

    def __init__(self, repo: str, cause: Exception | None = None) -> None:
        msg = f"Failed to download ONNX model from '{repo}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)
        self.repo = repo
        self.cause = cause


class ModelNotFoundError(EncoderError):
    """Raised when no ONNX model file is found in the local cache."""

    def __init__(self, search_paths: list[str]) -> None:
        paths = ", ".join(search_paths)
        super().__init__(
            f"No ONNX model file found. Searched: {paths}. "
            "Run: optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 <dir>"
        )
        self.search_paths = search_paths


class TokenizerNotFoundError(EncoderError):
    """Raised when tokenizer.json is missing from the model cache."""

    def __init__(self, path: str) -> None:
        super().__init__(f"tokenizer.json not found at '{path}'")
        self.path = path


# ── Taxonomy ───────────────────────────────────────────────────────

class TaxonomyError(PrismLangError):
    """Raised for invalid taxonomy configuration or usage."""


class DuplicateCategoryError(TaxonomyError):
    """Raised when two categories share the same slug."""

    def __init__(self, slug: str) -> None:
        super().__init__(f"Duplicate category slug: '{slug}'. All slugs must be unique.")
        self.slug = slug


class UnknownCategoryError(TaxonomyError):
    """Raised when a requested category slug is not in the taxonomy."""

    def __init__(self, slug: str, available: list[str]) -> None:
        super().__init__(
            f"Unknown category slug: '{slug}'. "
            f"Available: {available}"
        )
        self.slug = slug
        self.available = available


class EmptyTaxonomyError(TaxonomyError):
    """Raised when a TaxonomyConfig is created with no categories."""

    def __init__(self) -> None:
        super().__init__("TaxonomyConfig requires at least one Category.")


# ── Projector ──────────────────────────────────────────────────────

class ProjectionError(PrismLangError):
    """Raised when the PrismProjector cannot produce a valid vector."""


class ZeroVectorError(ProjectionError):
    """Raised when the encoder returns a near-zero vector (likely empty input)."""

    def __init__(self, text: str) -> None:
        preview = text[:50] + "..." if len(text) > 50 else text
        super().__init__(
            f"Encoder produced a near-zero vector for input: {preview!r}. "
            "Ensure input text is non-empty and contains meaningful content."
        )
        self.text = text


class DimensionMismatchError(ProjectionError):
    """Raised when the JL matrix and encoder output dimensions are incompatible."""

    def __init__(self, matrix_cols: int, vector_dim: int) -> None:
        super().__init__(
            f"JL matrix expects input dimension {matrix_cols}, "
            f"but encoder produced dimension {vector_dim}."
        )


# ── Checkpointer ───────────────────────────────────────────────────

class CheckpointerError(PrismLangError):
    """Raised when checkpoint persistence or retrieval fails."""


class CheckpointerConnectionError(CheckpointerError):
    """Raised when the checkpointer cannot connect to its backend."""

    def __init__(self, backend: str, cause: Exception | None = None) -> None:
        msg = f"Cannot connect to checkpointer backend '{backend}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)
        self.backend = backend
        self.cause = cause


class CheckpointerSchemaError(CheckpointerError):
    """Raised when the checkpoint schema cannot be created or is incompatible."""


# ── Tenant ─────────────────────────────────────────────────────────

class TenantError(PrismLangError):
    """Raised for tenant configuration or isolation violations."""


class MissingTenantError(TenantError):
    """Raised when tenant_id is missing from the PrismState."""

    def __init__(self) -> None:
        super().__init__(
            "PrismState is missing 'tenant_id'. "
            "Initialise state with: {'tenant_id': 'your-org-id', 'prism_sequence': [], 'raw_output': ''}"
        )
