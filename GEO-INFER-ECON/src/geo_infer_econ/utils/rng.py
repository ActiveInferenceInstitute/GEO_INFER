"""Deterministic-by-default RNG resolution for geo_infer_econ.

Mirrors the repo-wide ``resolve_rng`` pattern (see GEO-INFER-SPM/MATH/RISK/
BAYES ``utils/rng.py``) with one deliberate difference: a ``None`` seed
resolves to a *fixed* seed so library code paths are reproducible by default.
Callers that want fresh entropy pass an explicit unseeded
``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["SeedLike", "resolve_rng", "resolve_optional_rng"]

SeedLike = Union[
    None,
    int,
    np.integer,
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
    np.random.RandomState,
]

#: Seed used when a caller does not supply one, so that library paths are
#: deterministic by default.
DEFAULT_SEED = 0


def resolve_rng(seed: SeedLike = None) -> np.random.Generator:
    """Resolve ``seed`` into a :class:`numpy.random.Generator`.

    - A ``Generator`` is returned unchanged.
    - A legacy ``RandomState`` is bridged to a fresh, isolated ``Generator``.
    - Any other seed form (``int``, ``SeedSequence``, ``BitGenerator``) is
      passed to :func:`numpy.random.default_rng`.
    - ``None`` resolves to ``default_rng(DEFAULT_SEED)`` — deterministic by
      default, unlike the SPM/MATH/RISK/BAYES variants which draw fresh
      entropy for ``None``.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(int(seed.randint(0, 2**31 - 1)))
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def resolve_optional_rng(
    rng: SeedLike | np.random.Generator | None,
) -> np.random.Generator | None:
    """Return ``None`` when ``rng`` is ``None``, else :func:`resolve_rng`.

    For call sites where absence of an RNG means "take the deterministic
    non-stochastic branch" rather than "use the default seed".
    """
    return None if rng is None else resolve_rng(rng)
