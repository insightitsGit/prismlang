"""Tests for the ONNX encoder.

These tests use a lightweight mock to avoid downloading the model in CI.
To run against the real ONNX model, set PRISMLANG_USE_REAL_ENCODER=1.
"""

import os
import numpy as np
import pytest

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
