"""Tests for PrismProjector — the core math engine."""

import hashlib
import numpy as np
import pytest

from prismlang import Category, PrismProjector, TaxonomyConfig


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture()
def taxonomy():
    return TaxonomyConfig(
        categories=[
            Category("risk", "Risk", ["risk", "exposure", "volatility", "default"]),
            Category("market", "Market", ["market", "price", "equity", "bond"]),
            Category("compliance", "Compliance", ["compliance", "regulatory", "audit"]),
        ],
        alpha=0.3,
    )


@pytest.fixture()
def mock_encoder(monkeypatch):
    """Replace the encoder with a deterministic mock."""
    import prismlang.encoder as enc_mod

    def _encode(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)

    def _encode_batch(texts):
        return np.stack([_encode(t) for t in texts])

    monkeypatch.setattr(enc_mod, "encode", _encode)
    monkeypatch.setattr(enc_mod, "encode_batch", _encode_batch)


@pytest.fixture()
def projector(taxonomy, mock_encoder):
    return PrismProjector(taxonomy=taxonomy, tenant_id="test-tenant", k=64)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

def test_project_returns_tuple(projector):
    slug, vec, chain = projector.project("the credit risk exposure is high")
    assert isinstance(slug, str)
    assert isinstance(vec, np.ndarray)
    assert isinstance(chain, list)


def test_project_vector_shape(projector):
    _, vec, _ = projector.project("equity price movement")
    assert vec.shape == (64,)


def test_project_vector_unit_norm(projector):
    _, vec, _ = projector.project("regulatory compliance audit")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Output vector not unit-normalised: {norm}"


def test_project_dtype(projector):
    _, vec, _ = projector.project("any text")
    assert vec.dtype == np.float32


def test_project_deterministic(projector):
    text = "volatility in emerging markets"
    _, v1, _ = projector.project(text)
    _, v2, _ = projector.project(text)
    np.testing.assert_array_equal(v1, v2)


def test_rule_chain_contains_tenant(projector):
    _, _, chain = projector.project("some financial text")
    full_chain = " ".join(chain)
    assert "test-tenant" in full_chain


def test_rule_chain_contains_category(taxonomy, mock_encoder):
    proj = PrismProjector(taxonomy=taxonomy, tenant_id="tenant-x", k=32)
    _, _, chain = proj.project("credit default risk exposure volatility")
    # The inferred category should appear in the chain
    category_steps = [s for s in chain if "category_inference" in s]
    assert len(category_steps) == 1
    assert "risk" in category_steps[0]


def test_jl_matrix_seeded_by_tenant(taxonomy, mock_encoder):
    proj_a = PrismProjector(taxonomy=taxonomy, tenant_id="tenant-a", k=64)
    proj_b = PrismProjector(taxonomy=taxonomy, tenant_id="tenant-b", k=64)
    # Matrices must differ
    assert not np.allclose(proj_a._P, proj_b._P)


def test_k_dimension_respected(taxonomy, mock_encoder):
    for k in [16, 32, 64, 128]:
        proj = PrismProjector(taxonomy=taxonomy, tenant_id="t", k=k)
        _, vec, _ = proj.project("test")
        assert vec.shape == (k,), f"Expected k={k}, got {vec.shape}"


def test_project_batch_consistent(projector):
    texts = ["risk exposure default", "market equity price", "compliance regulatory"]
    batch = projector.project_batch(texts)
    assert len(batch) == 3
    for text, (slug, vec, chain) in zip(texts, batch):
        _, vec_single, _ = projector.project(text)
        np.testing.assert_allclose(vec, vec_single, atol=1e-6)


def test_matrix_fingerprint_stable(projector):
    fp1 = projector.matrix_fingerprint()
    fp2 = projector.matrix_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 16


def test_seed_derived_from_sha256():
    tenant = "acme-corp"
    expected_seed = int(hashlib.sha256(tenant.encode()).hexdigest(), 16) % (2**32)
    from prismlang.projector import _seed_from_tenant
    assert _seed_from_tenant(tenant) == expected_seed
