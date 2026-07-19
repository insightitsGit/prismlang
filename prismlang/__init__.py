"""PrismLang — Deterministic Vector Language Protocol for Multi-Agent AI Orchestration.

Public API
----------
    from prismlang import (
        PrismState,
        PrismEnvelope,
        PrismProjector,
        TaxonomyConfig,
        Category,
        prism_node,
        BoundaryTranslator,
        JsonFileCheckpointer,
        PostgresCheckpointer,
        model_id,
        get_session,
    )
"""

from .config import DEFAULT_ALPHA, DEFAULT_K, EMBED_DIM
from .envelope import PrismEnvelope
from .exceptions import (
    PrismLangError,
    EncoderError,
    ModelDownloadError,
    ModelNotFoundError,
    TokenizerNotFoundError,
    TaxonomyError,
    DuplicateCategoryError,
    UnknownCategoryError,
    EmptyTaxonomyError,
    ProjectionError,
    ZeroVectorError,
    DimensionMismatchError,
    CheckpointerError,
    CheckpointerConnectionError,
    CheckpointerSchemaError,
    TenantError,
    MissingTenantError,
)
from . import encoder
from .encoder import get_session, model_id
from .state import PrismState
from .taxonomy import Category, TaxonomyConfig
from .projector import PrismProjector
from .middleware import prism_node, async_prism_node
from .translator import BoundaryTranslator
from .checkpointer import (
    JsonFileCheckpointer,
    PostgresCheckpointer,
    AsyncJsonFileCheckpointer,
    AsyncPostgresCheckpointer,
)

__version__ = "0.1.2"
__author__ = "Amin Parva / Insight IT Solutions LLC"

__all__ = [
    "PrismState",
    "PrismEnvelope",
    "PrismProjector",
    "TaxonomyConfig",
    "Category",
    "prism_node",
    "BoundaryTranslator",
    "JsonFileCheckpointer",
    "PostgresCheckpointer",
    "AsyncJsonFileCheckpointer",
    "AsyncPostgresCheckpointer",
    "async_prism_node",
    "encoder",
    "get_session",
    "model_id",
    "DEFAULT_ALPHA",
    "DEFAULT_K",
    "EMBED_DIM",
    # Exceptions
    "PrismLangError",
    "EncoderError",
    "ModelDownloadError",
    "ModelNotFoundError",
    "TokenizerNotFoundError",
    "TaxonomyError",
    "DuplicateCategoryError",
    "UnknownCategoryError",
    "EmptyTaxonomyError",
    "ProjectionError",
    "ZeroVectorError",
    "DimensionMismatchError",
    "CheckpointerError",
    "CheckpointerConnectionError",
    "CheckpointerSchemaError",
    "TenantError",
    "MissingTenantError",
]
