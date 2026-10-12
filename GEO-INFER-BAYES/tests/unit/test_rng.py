"""Tests for the BAYES random-number-generator resolver.

These pin the contract every sampler in GEO-INFER-BAYES relies on: a seed
always becomes an explicit ``Generator``, the process-wide ``numpy.random``
singleton is never read or advanced, and ``0`` is a real seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo_infer_bayes.utils.rng import derive_int_seed, resolve_rng, spawn_rng


class TestResolveRng:
    def test_none_gives_an_independent_generator(self) -> None:
        """``None`` never returns the numpy.random module or a shared object."""
        first, second = resolve_rng(), resolve_rng()
        assert isinstance(first, np.random.Generator)
        assert first is not second
        assert first.random() != second.random()

    def test_none_does_not_read_the_global_stream(self) -> None:
        """Seeding numpy.random cannot make an unseeded call reproducible."""
        np.random.seed(7)
        first = resolve_rng().random()
        np.random.seed(7)
        second = resolve_rng().random()
        assert first != second

    def test_none_does_not_advance_the_global_stream(self) -> None:
        np.random.seed(7)
        expected = np.random.random()
        np.random.seed(7)
        resolve_rng().random()
        assert np.random.random() == expected

    @pytest.mark.parametrize("seed", [0, 1, 12345, np.int64(9)])
    def test_equal_int_seeds_replay(self, seed: object) -> None:
        """Zero is a real seed, not a falsy no-op."""
        assert resolve_rng(seed).random() == resolve_rng(seed).random()

    def test_distinct_int_seeds_diverge(self) -> None:
        assert resolve_rng(1).random() != resolve_rng(2).random()

    def test_a_generator_is_returned_unchanged(self) -> None:
        """Threading one generator through a pipeline must not restart it."""
        generator = np.random.default_rng(3)
        assert resolve_rng(generator) is generator

    def test_seed_sequence_is_honoured(self) -> None:
        entropy = np.random.SeedSequence(42)
        a = resolve_rng(np.random.SeedSequence(42)).random()
        b = resolve_rng(entropy).random()
        assert a == b

    def test_bit_generator_is_honoured(self) -> None:
        a = resolve_rng(np.random.PCG64(5)).random()
        b = resolve_rng(np.random.PCG64(5)).random()
        assert a == b

    @pytest.mark.parametrize("legacy", [np.random.RandomState(4), np.random])
    def test_legacy_numpy_random_objects_are_rejected(self, legacy: object) -> None:
        """RandomState and the numpy.random module are not seed forms."""
        with pytest.raises(TypeError, match="seed must be"):
            resolve_rng(legacy)

    @pytest.mark.parametrize("bad", ["abc", 1.5, [1, 2], {"seed": 1}])
    def test_unusable_seeds_are_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError, match="seed must be"):
            resolve_rng(bad)


class TestSpawnRng:
    def test_children_are_reproducible_from_the_parent_seed(self) -> None:
        a = [g.random() for g in spawn_rng(4, 3)]
        b = [g.random() for g in spawn_rng(4, 3)]
        assert a == b

    def test_children_are_mutually_distinct(self) -> None:
        draws = [g.random() for g in spawn_rng(4, 5)]
        assert len(set(draws)) == 5

    def test_children_differ_from_a_plainly_offset_seed(self) -> None:
        """Spawning is not the same as seeding with seed, seed+1, ..."""
        spawned = [g.random() for g in spawn_rng(4, 3)]
        offset = [resolve_rng(4 + i).random() for i in range(3)]
        assert spawned != offset

    def test_zero_children_is_an_empty_list(self) -> None:
        assert spawn_rng(1, 0) == []

    def test_negative_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            spawn_rng(1, -1)

    def test_an_unseeded_parent_still_spawns_distinct_children(self) -> None:
        draws = [g.random() for g in spawn_rng(None, 4)]
        assert len(set(draws)) == 4


class TestDeriveIntSeed:
    def test_is_reproducible_for_an_int_seed(self) -> None:
        assert derive_int_seed(9) == derive_int_seed(9)

    def test_lies_in_the_scikit_learn_accepted_range(self) -> None:
        for seed in (0, 1, 999):
            value = derive_int_seed(seed)
            assert isinstance(value, int)
            assert 0 <= value < 2**32

    def test_distinct_seeds_give_distinct_values(self) -> None:
        assert derive_int_seed(1) != derive_int_seed(2)

    def test_a_generator_is_advanced_so_repeated_calls_differ(self) -> None:
        generator = np.random.default_rng(0)
        assert derive_int_seed(generator) != derive_int_seed(generator)

    def test_accepts_none(self) -> None:
        assert 0 <= derive_int_seed() < 2**32
