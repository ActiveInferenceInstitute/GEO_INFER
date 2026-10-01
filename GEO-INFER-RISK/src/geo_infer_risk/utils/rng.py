"""Explicit random-number-generator plumbing for GEO-INFER-RISK.

Catastrophe simulation, exposure sampling and Monte Carlo loss aggregation are
all stochastic, and a risk number is only defensible if the run behind it can
be replayed.

Every stochastic entry point resolves its ``seed`` / ``random_state`` argument
through :func:`resolve_rng`, so no code path touches the process-wide
``numpy.random`` singleton. This file shares one implementation with every
other ``geo_infer_*/utils/rng.py``; only this docstring and
:data:`DEFAULT_SEED` differ, and ``validate_repo_contracts.py`` enforces that.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

__all__ = [
    "DEFAULT_SEED",
    "SeedLike",
    "derive_int_seed",
    "resolve_optional_rng",
    "resolve_rng",
    "spawn_rng",
]

#: Anything accepted by :func:`resolve_rng`.
SeedLike: TypeAlias = (
    int
    | np.integer
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | None
)

#: ``None`` seeds draw fresh OS entropy; pass an ``int`` when replay matters.
DEFAULT_SEED: int | None = None


def resolve_rng(seed: SeedLike = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for ``seed``.

    Parameters
    ----------
    seed:
        * ``None`` -- ``default_rng(DEFAULT_SEED)``.
        * ``int`` / ``numpy.integer`` -- a deterministic generator; equal
          values give equal streams.
        * :class:`numpy.random.SeedSequence` or
          :class:`numpy.random.BitGenerator` -- used directly, the supported
          way to hand out independent child streams.
        * :class:`numpy.random.Generator` -- returned unchanged so one
          generator can be threaded through a pipeline.

    Returns
    -------
    numpy.random.Generator
        A generator instance, never the ``numpy.random`` module.

    Raises
    ------
    TypeError
        If ``seed`` cannot produce a generator (including the legacy
        ``numpy.random.RandomState`` and the ``numpy.random`` module).

    Examples
    --------
    >>> resolve_rng(11).integers(0, 10) == resolve_rng(11).integers(0, 10)
    True
    >>> shared = np.random.default_rng(0)
    >>> resolve_rng(shared) is shared
    True
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng(DEFAULT_SEED)
    if isinstance(
        seed, (int, np.integer, np.random.SeedSequence, np.random.BitGenerator)
    ):
        return np.random.default_rng(seed)
    raise TypeError(
        "seed must be None, an int, a SeedSequence, a BitGenerator or a "
        f"Generator; got {type(seed).__name__}"
    )


def resolve_optional_rng(seed: SeedLike) -> np.random.Generator | None:
    """Return ``None`` for ``None``, else :func:`resolve_rng`.

    For call sites where an absent RNG selects a deterministic,
    non-stochastic branch rather than the default seed.
    """
    return None if seed is None else resolve_rng(seed)


def spawn_rng(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent generators derived from ``seed``.

    Uses :meth:`numpy.random.SeedSequence.spawn`, the supported mechanism for
    splitting one seed into non-overlapping streams (unlike ``seed``,
    ``seed + 1``, ... which carries no independence guarantee).

    Raises
    ------
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    entropy = int(resolve_rng(seed).integers(0, 2**63 - 1, dtype=np.int64))
    return [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(entropy).spawn(n)
    ]


def derive_int_seed(seed: SeedLike = None) -> int:
    """Derive a plain ``int`` seed in ``[0, 2**32)`` from any seed-like value.

    Use at boundaries (scikit-learn ``random_state``, some SciPy routines)
    that accept only integers. Passing a ``Generator`` advances it, so
    repeated calls on one generator give distinct downstream seeds.
    """
    return int(resolve_rng(seed).integers(0, 2**32, dtype=np.int64))
