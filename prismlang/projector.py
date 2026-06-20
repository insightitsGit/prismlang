"""PrismProjector — core mathematical engine of PrismLang.

Implements the two-step protocol from the paper:
  1. Spherical Blend  (eq. 1): v' = normalize((1-α)·v + α·‖v‖·e_i)
  2. JL Reduction     (eq. 2): p  = normalize(P · v')

The Johnson-Lindenstrauss matrix P is seeded deterministically from the
SHA-256 hash of tenant_id, giving each tenant a unique projection space.
A vector intercepted from one tenant's traffic is geometrically meaningless
to a model operating under a different tenant's matrix.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np

from .config import DEFAULT_ALPHA, DEFAULT_K, EMBED_DIM
from .exceptions import DimensionMismatchError, ZeroVectorError
from .taxonomy import TaxonomyConfig


def _seed_from_tenant(tenant_id: str) -> int:
    digest = hashlib.sha256(tenant_id.encode()).hexdigest()
    return int(digest, 16) % (2**32)


def _build_jl_matrix(k: int, d: int, seed: int) -> np.ndarray:
    """Return a (k, d) row-normalised Gaussian random matrix."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((k, d)).astype(np.float32)
    norms = np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
    return P / norms


class PrismProjector:
    """Encodes text into a k-dimensional PrismLang vector for a specific tenant."""

    def __init__(
        self,
        taxonomy: TaxonomyConfig,
        tenant_id: str,
        k: int = DEFAULT_K,
        alpha: float | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.tenant_id = tenant_id
        self.k = k
        self.alpha = alpha if alpha is not None else taxonomy.alpha

        self._seed = _seed_from_tenant(tenant_id)
        self._P = _build_jl_matrix(k, EMBED_DIM, self._seed)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def project(self, text: str) -> Tuple[str, np.ndarray, List[str]]:
        """Project text → (category_slug, k-dim unit vector, audit rule_chain).

        The rule_chain is an ordered list of strings that fully describes every
        decision made during projection — allowing human overseers to trace the
        output vector back to its organisational taxonomy rule.
        """
        from . import encoder  # lazy to keep import graph clean

        # Step 0: embed
        if not text or not text.strip():
            raise ZeroVectorError(text)
        v = encoder.encode(text)                          # (384,) unit vector
        if v.shape[0] != self._P.shape[1]:
            raise DimensionMismatchError(self._P.shape[1], v.shape[0])

        # Step 1: category inference (deterministic keyword match)
        slug = self.taxonomy.infer_category(text)
        e_i = self.taxonomy.direction_vector(slug)        # (384,) unit vector

        # Step 2: spherical blend  →  v'
        raw_blend = (1.0 - self.alpha) * v + self.alpha * float(np.linalg.norm(v)) * e_i
        norm_blend = np.linalg.norm(raw_blend)
        v_prime = raw_blend / max(norm_blend, 1e-12)      # (384,) unit vector

        # Step 3: JL reduction  →  p
        p_raw = self._P @ v_prime                         # (k,)
        p_norm = np.linalg.norm(p_raw)
        p = p_raw / max(p_norm, 1e-12)                   # (k,) unit vector

        rule_chain = [
            f"text -> encoder(all-MiniLM-L6-v2, d={EMBED_DIM})",
            f"category_inference -> slug={slug!r}",
            f"spherical_blend(alpha={self.alpha:.3f}) -> v_prime",
            f"JL_reduction(seed=sha256({self.tenant_id!r}), k={self.k}) -> p",
        ]

        return slug, p.astype(np.float32), rule_chain

    def project_batch(self, texts: List[str]) -> List[Tuple[str, np.ndarray, List[str]]]:
        """Project multiple texts. Embeddings are batched for efficiency."""
        from . import encoder

        vectors = encoder.encode_batch(texts)             # (N, 384)
        results = []
        for text, v in zip(texts, vectors):
            slug = self.taxonomy.infer_category(text)
            e_i = self.taxonomy.direction_vector(slug)
            raw_blend = (1.0 - self.alpha) * v + self.alpha * float(np.linalg.norm(v)) * e_i
            norm_blend = np.linalg.norm(raw_blend)
            v_prime = raw_blend / max(norm_blend, 1e-12)
            p_raw = self._P @ v_prime
            p = p_raw / max(np.linalg.norm(p_raw), 1e-12)
            rule_chain = [
                f"text -> encoder(all-MiniLM-L6-v2, d={EMBED_DIM})",
                f"category_inference -> slug={slug!r}",
                f"spherical_blend(alpha={self.alpha:.3f}) -> v_prime",
                f"JL_reduction(seed=sha256({self.tenant_id!r}), k={self.k}) -> p",
            ]
            results.append((slug, p.astype(np.float32), rule_chain))
        return results

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def matrix_fingerprint(self) -> str:
        """Return a short hex fingerprint of the JL matrix for audit logs."""
        digest = hashlib.sha256(self._P.tobytes()).hexdigest()
        return digest[:16]

    def __repr__(self) -> str:
        return (
            f"PrismProjector(tenant={self.tenant_id!r}, k={self.k}, "
            f"alpha={self.alpha}, seed={self._seed}, "
            f"matrix_fp={self.matrix_fingerprint()!r})"
        )
