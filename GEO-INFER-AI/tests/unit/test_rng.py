"""Tests for the AI random-number-generator resolver.

These pin the contract every stochastic entry point in GEO-INFER-AI relies
on: a seed always becomes an explicit ``Generator``, and ``None`` resolves
to the fixed :data:`DEFAULT_SEED` — AI's deliberate difference from the
SPM/MATH/RISK/BAYES family, which draw fresh entropy for ``None``. Library
paths must therefore be reproducible by default while callers wanting fresh
entropy pass an explicit unseeded ``Generator``.
"""

from __future__ import annotations

import numpy as np
import pytest

import geo_infer_ai.utils as utils
from geo_infer_ai.utils import SeedLike, resolve_rng
from geo_infer_ai.utils.rng import DEFAULT_SEED


class TestResolveRngDeterministicByDefault:
    """``None`` resolves to a fixed seed — the deliberate family deviation."""

    def test_default_seed_is_zero(self) -> None:
        assert DEFAULT_SEED == 0

    def test_none_is_reproducible(self) -> None:
        """Two unseeded calls replay identically, unlike the sibling modules."""
        first = resolve_rng(None)
        second = resolve_rng(None)
        assert first.random() == second.random()

    def test_seeding_numpy_random_does_not_change_the_fixed_stream(self) -> None:
        """The fixed default is independent of the process-wide numpy state."""
        expected = resolve_rng(None).random()
        np.random.seed(7)
        assert resolve_rng(None).random() == expected

    def test_none_matches_default_rng_zero(self) -> None:
        assert (
            resolve_rng(None).random() == np.random.default_rng(DEFAULT_SEED).random()
        )

    def test_zero_is_a_real_seed_and_replays(self) -> None:
        """An explicit ``0`` agrees with the unseeded default, not a no-op."""
        assert resolve_rng(0).random() == resolve_rng(None).random()

    @pytest.mark.parametrize("seed", [0, 1, 12345, np.int64(9)])
    def test_equal_int_seeds_replay(self, seed: object) -> None:
        assert resolve_rng(seed).random() == resolve_rng(seed).random()

    def test_distinct_int_seeds_diverge(self) -> None:
        assert resolve_rng(1).random() != resolve_rng(2).random()


class TestResolveRngPassThrough:
    """An already-explicit generator must survive resolution unchanged."""

    def test_a_generator_is_returned_unchanged(self) -> None:
        """Threading one generator through a pipeline must not restart it."""
        generator = np.random.default_rng(3)
        assert resolve_rng(generator) is generator

    def test_seeded_generator_is_not_reset(self) -> None:
        """Advancing a passed generator must advance the resolved one too."""
        generator = np.random.default_rng(3)
        expected = generator.random()
        assert resolve_rng(generator).random() != expected


class TestResolveRngSeedForms:
    def test_seed_sequence_is_honoured(self) -> None:
        entropy = np.random.SeedSequence(42)
        a = resolve_rng(entropy).random()
        b = resolve_rng(np.random.SeedSequence(42)).random()
        assert a == b

    def test_bit_generator_is_honoured(self) -> None:
        a = resolve_rng(np.random.PCG64(5)).random()
        b = resolve_rng(np.random.PCG64(5)).random()
        assert a == b


class TestResolveRngRejectsLegacyState:
    @pytest.mark.parametrize("legacy", [np.random.RandomState(4), np.random])
    def test_legacy_numpy_random_objects_are_rejected(self, legacy: object) -> None:
        """RandomState and the numpy.random module are not seed forms."""
        with pytest.raises(TypeError, match="seed must be"):
            resolve_rng(legacy)


class TestUtilsExports:
    def test_resolve_rng_is_exported(self) -> None:
        assert utils.resolve_rng is resolve_rng

    def test_seed_like_is_exported(self) -> None:
        assert "SeedLike" in utils.__all__
        assert SeedLike is not None

    def test_no_dead_optional_rng_export(self) -> None:
        """``resolve_optional_rng`` was removed as a dead export: it must not
        reappear in the package surface without a live caller."""
        assert "resolve_optional_rng" not in utils.__all__
        assert not hasattr(utils, "resolve_optional_rng")
