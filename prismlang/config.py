"""Global constants for PrismLang."""

# ONNX encoder output dimension (all-MiniLM-L6-v2)
EMBED_DIM: int = 384

# Default Johnson-Lindenstrauss target dimension
DEFAULT_K: int = 64

# Spherical blend weight — how strongly the category direction pulls the vector
DEFAULT_ALPHA: float = 0.3

# HuggingFace model repo for the ONNX encoder
HF_MODEL_REPO: str = "sentence-transformers/all-MiniLM-L6-v2"

# Local cache directory for downloaded models
import os
MODEL_CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".prismlang", "models")
