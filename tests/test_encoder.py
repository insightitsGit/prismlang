"""Tests for the ONNX encoder.

These tests use a lightweight mock to avoid downloading the model in CI.
To run against the real ONNX model, set PRISMLANG_USE_REAL_ENCODER=1.
"""

import os
import re

import numpy as np
import pytest

from prismlang.config import HF_MODEL_REPO
from prismlang.encoder import _compute_model_id, _read_revision

USE_REAL = os.environ.get("PRISMLANG_USE_REAL_ENCODER", "0") == "1"


# ------------------------------------------------------------------ #
# Mock encoder for fast unit tests                                     #
# ------------------------------------------------------------------ #

class _MockEncoder:
    """Deterministic mock that returns a seeded unit vector for each text."""
    DIM = 384

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_batch(self, texts):
        return np.stack([self.encode(t) for t in texts])


@pytest.fixture()
def enc(monkeypatch):
    if USE_REAL:
        from prismlang import encoder
        return encoder
    mock = _MockEncoder()
    import prismlang.encoder as enc_mod
    monkeypatch.setattr(enc_mod, "encode", mock.encode)
    monkeypatch.setattr(enc_mod, "encode_batch", mock.encode_batch)
    import prismlang.encoder as enc_mod2
    return enc_mod2


def test_encode_shape(enc):
    v = enc.encode("hello world")
    assert v.shape == (384,), f"Expected (384,), got {v.shape}"


def test_encode_unit_norm(enc):
    v = enc.encode("test sentence for normalisation")
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"Vector not unit-normalised: norm={norm}"


def test_encode_dtype(enc):
    v = enc.encode("dtype check")
    assert v.dtype == np.float32


def test_encode_batch_shape(enc):
    texts = ["sentence one", "sentence two", "sentence three"]
    vecs = enc.encode_batch(texts)
    assert vecs.shape == (3, 384)


def test_encode_batch_unit_norms(enc):
    texts = ["alpha", "beta", "gamma"]
    vecs = enc.encode_batch(texts)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_same_text_same_vector(enc):
    v1 = enc.encode("deterministic input")
    v2 = enc.encode("deterministic input")
    np.testing.assert_array_equal(v1, v2)


def test_different_texts_different_vectors(enc):
    v1 = enc.encode("risk management in finance")
    v2 = enc.encode("regulatory compliance requirements")
    assert not np.allclose(v1, v2), "Different texts should produce different vectors"


# ------------------------------------------------------------------ #
# model_id() — artifact identity                                       #
# ------------------------------------------------------------------ #

MODEL_ID_PATTERN = re.compile(rf"^{re.escape(HF_MODEL_REPO)}@[^:]+:[0-9a-f]{{12}}$")


def test_compute_model_id_format(tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"onnx-bytes-v1")
    mid = _compute_model_id(fake_model)
    assert MODEL_ID_PATTERN.match(mid), f"Unexpected model id format: {mid}"


def test_compute_model_id_stable_for_same_file(tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"onnx-bytes-v1")
    assert _compute_model_id(fake_model) == _compute_model_id(fake_model)


def test_compute_model_id_changes_when_model_file_changes(tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"onnx-bytes-v1")
    id_before = _compute_model_id(fake_model)
    fake_model.write_bytes(b"onnx-bytes-v2-CHANGED")
    id_after = _compute_model_id(fake_model)
    assert id_before != id_after, "model_id must change when the model artifact changes"
    # Repo prefix stays constant; only the content hash differs
    assert id_before.rsplit(":", 1)[0] == id_after.rsplit(":", 1)[0]


def test_read_revision_from_hf_metadata(tmp_path):
    model_dir = tmp_path / "all-MiniLM-L6-v2"
    onnx_path = model_dir / "onnx" / "model.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_bytes(b"onnx-bytes")
    meta = model_dir / ".cache" / "huggingface" / "download" / "onnx" / "model.onnx.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text("abc123def456\nsomesha\n1700000000.0\n", encoding="utf-8")
    assert _read_revision(onnx_path) == "abc123def456"
    assert "@abc123def456:" in _compute_model_id(onnx_path)


def test_read_revision_falls_back_to_local(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    assert _read_revision(onnx_path) == "local"


def test_model_id_cached_and_stable_across_calls(monkeypatch):
    """Public model_id() returns the value computed at session init, unchanged."""
    import prismlang.encoder as enc_mod

    sentinel_session = object()
    cached = f"{HF_MODEL_REPO}@abc123def456:0123456789ab"
    monkeypatch.setattr(enc_mod, "_session", sentinel_session)
    monkeypatch.setattr(enc_mod, "_tokenizer", object())
    monkeypatch.setattr(enc_mod, "_model_id", cached)

    first = enc_mod.model_id()
    second = enc_mod.model_id()
    assert first == cached
    assert first == second


@pytest.mark.skipif(not USE_REAL, reason="requires real ONNX model (PRISMLANG_USE_REAL_ENCODER=1)")
def test_model_id_real_model():
    from prismlang import encoder
    mid1 = encoder.model_id()
    mid2 = encoder.model_id()
    assert mid1 == mid2
    assert MODEL_ID_PATTERN.match(mid1), f"Unexpected model id format: {mid1}"


# ------------------------------------------------------------------ #
# get_session() — shared session identity                              #
# ------------------------------------------------------------------ #

def test_get_session_returns_singleton(monkeypatch):
    """All encode paths and get_session() share one process-wide session object."""
    import prismlang.encoder as enc_mod

    sentinel_session = object()
    monkeypatch.setattr(enc_mod, "_session", sentinel_session)
    monkeypatch.setattr(enc_mod, "_tokenizer", object())
    monkeypatch.setattr(enc_mod, "_model_id", "repo@rev:0123456789ab")

    # Public accessor and the internal session used by encode/encode_batch
    assert enc_mod.get_session() is sentinel_session
    assert enc_mod._get_session()[0] is sentinel_session
    # Repeated calls hand back the identical object — no second load
    assert enc_mod.get_session() is enc_mod.get_session()


@pytest.mark.skipif(not USE_REAL, reason="requires real ONNX model (PRISMLANG_USE_REAL_ENCODER=1)")
def test_get_session_identity_real_encode_paths():
    """With the real model: encode() and encode_batch() run on the session
    returned by get_session(), and only one session exists."""
    import prismlang.encoder as enc_mod

    session = enc_mod.get_session()
    v1 = enc_mod.encode("session identity check")
    v2 = enc_mod.encode_batch(["session identity check"])[0]
    assert enc_mod.get_session() is session
    assert enc_mod._get_session()[0] is session
    np.testing.assert_array_equal(v1, v2)


# ------------------------------------------------------------------ #
# Public API surface                                                   #
# ------------------------------------------------------------------ #

def test_top_level_exports():
    import prismlang
    import prismlang.encoder as enc_mod

    assert prismlang.model_id is enc_mod.model_id
    assert prismlang.get_session is enc_mod.get_session
    assert prismlang.encoder is enc_mod
    assert "model_id" in prismlang.__all__
    assert "get_session" in prismlang.__all__
    assert "encoder" in prismlang.__all__
