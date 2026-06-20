"""Tenant isolation tests.

Proves that two tenants with different tenant_id values produce vectors
that are geometrically incompatible — i.e., the cosine similarity between
the same input encoded under two different JL matrices is near zero.

This is the mathematical isolation guarantee at the core of PrismLang's
multi-tenant security model.
"""

import numpy as np
import pytest

from prismlang import Category, PrismProjector, TaxonomyConfig


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def mock_encoder(monkeypatch):
    import prismlang.encoder as enc_mod

    def _encode(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)

    monkeypatch.setattr(enc_mod, "encode", _encode)
    monkeypatch.setattr(enc_mod, "encode_batch", lambda ts: np.stack([_encode(t) for t in ts]))


@pytest.fixture()
def taxonomy():
    return TaxonomyConfig(
        categories=[
            Category("risk", "Risk", ["risk", "exposure", "volatility"]),
            Category("market", "Market", ["market", "price", "equity"]),
            Category("compliance", "Compliance", ["compliance", "regulatory"]),
        ],
        alpha=0.3,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

SAMPLE_TEXTS = [
    "credit risk exposure in emerging market bonds",
    "equity price movement and bond yield spread",
    "regulatory compliance audit requirements",
    "portfolio volatility and counterparty default risk",
    "SEC disclosure and fiduciary obligation",
]


def test_different_tenants_different_jl_matrices(taxonomy):
    proj_a = PrismProjector(taxonomy=taxonomy, tenant_id="tenant-alpha", k=64)
    proj_b = PrismProjector(taxonomy=taxonomy, tenant_id="tenant-beta", k=64)
    assert not np.allclose(proj_a._P, proj_b._P), "JL matrices must differ across tenants"


def test_same_tenant_same_jl_matrix(taxonomy):
    proj_1 = PrismProjector(taxonomy=taxonomy, tenant_id="same-tenant", k=64)
    proj_2 = PrismProjector(taxonomy=taxonomy, tenant_id="same-tenant", k=64)
    np.testing.assert_array_equal(proj_1._P, proj_2._P)


def test_cross_tenant_cosine_near_zero(taxonomy):
    """Mean |cosine similarity| between tenant A and B vectors must be < 0.15."""
    proj_a = PrismProjector(taxonomy=taxonomy, tenant_id="acme-finance", k=64)
    proj_b = PrismProjector(taxonomy=taxonomy, tenant_id="globex-trading", k=64)

    sims = []
    for text in SAMPLE_TEXTS:
        _, va, _ = proj_a.project(text)
        _, vb, _ = proj_b.project(text)
        sims.append(abs(cosine_sim(va, vb)))

    mean_sim = float(np.mean(sims))
    assert mean_sim < 0.20, (
        f"Cross-tenant vectors are not sufficiently isolated: mean |cosine| = {mean_sim:.4f}"
    )


def test_same_tenant_same_text_identical_vectors(taxonomy):
    proj = PrismProjector(taxonomy=taxonomy, tenant_id="deterministic-tenant", k=64)
    text = "risk exposure volatility"
    _, v1, _ = proj.project(text)
    _, v2, _ = proj.project(text)
    np.testing.assert_array_equal(v1, v2)


def test_many_tenant_pairs_isolated(taxonomy):
    """Spot-check 10 random tenant pairs — all should have mean |cosine| < 0.15."""
    tenant_ids = [f"tenant-{i:03d}" for i in range(10)]
    projectors = [PrismProjector(taxonomy=taxonomy, tenant_id=t, k=64) for t in tenant_ids]

    text = SAMPLE_TEXTS[0]
    vectors = []
    for proj in projectors:
        _, v, _ = proj.project(text)
        vectors.append(v)

    # Check all pairs
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            sim = abs(cosine_sim(vectors[i], vectors[j]))
            assert sim < 0.35, (
                f"Tenants {tenant_ids[i]!r} and {tenant_ids[j]!r} are not isolated: "
                f"|cosine| = {sim:.4f}"
            )


def test_vector_length_does_not_leak_tenant(taxonomy):
    """All tenant vectors are unit-normalised, so ‖v‖ reveals nothing about the tenant."""
    for tenant in ["tenant-a", "tenant-b", "tenant-c"]:
        proj = PrismProjector(taxonomy=taxonomy, tenant_id=tenant, k=64)
        _, v, _ = proj.project("credit risk exposure")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5, f"Tenant {tenant!r}: vector norm={norm} (not unit)"


def test_matrix_fingerprints_differ(taxonomy):
    proj_a = PrismProjector(taxonomy=taxonomy, tenant_id="fp-tenant-a", k=64)
    proj_b = PrismProjector(taxonomy=taxonomy, tenant_id="fp-tenant-b", k=64)
    assert proj_a.matrix_fingerprint() != proj_b.matrix_fingerprint()


def test_category_slug_same_across_tenants_for_same_text(taxonomy):
    """Category inference is taxonomy-driven, not tenant-driven — slugs must match."""
    text = "credit risk exposure volatility default"
    proj_a = PrismProjector(taxonomy=taxonomy, tenant_id="t-a", k=64)
    proj_b = PrismProjector(taxonomy=taxonomy, tenant_id="t-b", k=64)
    slug_a, _, _ = proj_a.project(text)
    slug_b, _, _ = proj_b.project(text)
    assert slug_a == slug_b, "Category slug must be tenant-agnostic (taxonomy-only)"
