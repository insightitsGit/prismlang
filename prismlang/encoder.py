"""ONNX-based text encoder using all-MiniLM-L6-v2.

Downloads the model from HuggingFace on first use and caches it locally.
Produces 384-d L2-normalised unit vectors — no external API required.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import EMBED_DIM, HF_MODEL_REPO, MODEL_CACHE_DIR
from .exceptions import ModelDownloadError, ModelNotFoundError, TokenizerNotFoundError, EncoderError

_lock = threading.Lock()
_session = None
_tokenizer = None


def _load() -> None:
    global _session, _tokenizer

    if _session is not None:
        return

    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download

    cache_dir = Path(MODEL_CACHE_DIR)
    model_dir = cache_dir / "all-MiniLM-L6-v2"

    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                local_dir=str(model_dir),
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
            )
        except Exception as exc:
            raise ModelDownloadError(HF_MODEL_REPO, cause=exc) from exc

    onnx_candidates = [
        model_dir / "onnx" / "model.onnx",
        model_dir / "model.onnx",
    ]
    onnx_path = next((p for p in onnx_candidates if p.exists()), None)
    if onnx_path is None:
        raise ModelNotFoundError([str(p) for p in onnx_candidates])

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise TokenizerNotFoundError(str(tokenizer_path))

    try:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        _session = ort.InferenceSession(str(onnx_path), sess_options=opts)
        _tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise EncoderError(f"Failed to initialise ONNX session: {exc}") from exc
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    _tokenizer.enable_truncation(max_length=128)


def _get_session():
    with _lock:
        _load()
    return _session, _tokenizer


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pool over non-padding tokens."""
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    return x / norms


def encode_batch(texts: Sequence[str]) -> np.ndarray:
    """Encode a list of strings → (N, 384) float32 unit vectors."""
    session, tokenizer = _get_session()

    encodings = tokenizer.encode_batch(list(texts))
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    # outputs[0] is the last hidden state: (N, seq_len, 384)
    token_embeddings = outputs[0]
    pooled = _mean_pool(token_embeddings, attention_mask)
    return _l2_normalize(pooled).astype(np.float32)


def encode(text: str) -> np.ndarray:
    """Encode a single string → (384,) float32 unit vector."""
    return encode_batch([text])[0]


async def async_encode(text: str) -> np.ndarray:
    """Async version of encode — runs ONNX inference in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode, text)


async def async_encode_batch(texts: Sequence[str]) -> np.ndarray:
    """Async version of encode_batch — runs ONNX inference in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode_batch, texts)
