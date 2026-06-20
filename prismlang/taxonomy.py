"""Taxonomy definition and category direction vectors (e_i).

A TaxonomyConfig holds a set of named categories, each with a keyword list.
At projection time, the encoder embeds those keywords once (cached) and
averages them to produce the category's direction vector in semantic space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .exceptions import DuplicateCategoryError, EmptyTaxonomyError, UnknownCategoryError


@dataclass
class Category:
    slug: str
    label: str
    keywords: List[str]


class TaxonomyConfig:
    def __init__(self, categories: List[Category], alpha: float = 0.3) -> None:
        if not categories:
            raise EmptyTaxonomyError()
        slugs = [c.slug for c in categories]
        seen: set[str] = set()
        for slug in slugs:
            if slug in seen:
                raise DuplicateCategoryError(slug)
            seen.add(slug)

        self.categories = categories
        self.alpha = alpha
        self._by_slug: Dict[str, Category] = {c.slug: c for c in categories}
        self._direction_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Direction vectors                                                    #
    # ------------------------------------------------------------------ #

    def direction_vector(self, slug: str) -> np.ndarray:
        """Return the 384-d direction unit vector e_i for the given category slug.

        Built lazily on first call and cached — the encoder is only imported
        here to avoid a circular import at module load time.
        """
        if slug not in self._direction_cache:
            from . import encoder  # lazy import

            cat = self._by_slug.get(slug)
            if cat is None:
                raise UnknownCategoryError(slug, self.slugs)

            vecs = encoder.encode_batch(cat.keywords)  # (K, 384)
            mean = vecs.mean(axis=0)
            norm = np.linalg.norm(mean)
            self._direction_cache[slug] = mean / max(norm, 1e-12)

        return self._direction_cache[slug]

    def warm_up(self) -> None:
        """Pre-compute all direction vectors (call once at startup to avoid latency on first request)."""
        for cat in self.categories:
            self.direction_vector(cat.slug)

    # ------------------------------------------------------------------ #
    # Category inference (deterministic keyword match)                     #
    # ------------------------------------------------------------------ #

    def infer_category(self, text: str) -> str:
        """Return the slug of the best-matching category via keyword hit count.

        Falls back to the first category when no keywords match — ensuring
        the projection is always deterministic even for out-of-vocabulary text.
        """
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for cat in self.categories:
            scores[cat.slug] = sum(kw.lower() in text_lower for kw in cat.keywords)

        best = max(scores, key=lambda s: scores[s])
        # If all scores are 0 return the first category (deterministic default)
        if scores[best] == 0:
            return self.categories[0].slug
        return best

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @property
    def slugs(self) -> List[str]:
        return [c.slug for c in self.categories]

    def __repr__(self) -> str:
        return f"TaxonomyConfig(categories={self.slugs}, alpha={self.alpha})"
