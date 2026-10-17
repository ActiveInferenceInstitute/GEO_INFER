"""Native computation cache contracts for scientific and concurrent callers."""

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from geo_infer_math.utils.caching import ComputationCache, cache_result


def test_default_lifetime_evicts_without_crashing_and_tracks_use():
    calls = []

    @cache_result(maxsize=2)
    def square(value):
        calls.append(value)
        return value * value

    assert [square(value) for value in [1, 2, 1, 3, 1, 2]] == [1, 4, 1, 9, 1, 4]
    assert calls == [1, 2, 3, 2]
    assert square.cache_info() == {"size": 2, "maxsize": 2, "hits": 2, "misses": 4}
    square.cache_clear()
    assert square.cache_info() == {"size": 0, "maxsize": 2, "hits": 0, "misses": 0}


def test_array_shape_dtype_and_mutation_remain_distinct():
    @cache_result()
    def describe(array):
        return array.shape, array.dtype.str, array.tolist()

    original = np.arange(4, dtype=np.int64)
    assert describe(original)[0] == (4,)
    assert describe(original.reshape(2, 2))[0] == (2, 2)
    assert describe(original.view(np.float64))[1] == np.dtype(np.float64).str
    original[0] = 9
    assert describe(original)[2] == [9, 1, 2, 3]
    assert describe.cache_info()["misses"] == 4


def test_object_array_keys_follow_contents_and_equal_independent_arrays():
    @cache_result()
    def first(array):
        return array[0]

    left = np.array(["left"], dtype=object)
    assert first(left) == "left"
    assert first(np.array(["left"], dtype=object)) == "left"
    left[0] = "right"
    assert first(left) == "right"
    assert first.cache_info()["hits"] == 1


def test_nested_arrays_and_keyword_order_have_structured_keys():
    @cache_result()
    def describe(*, data, label):
        return data["values"].shape, label

    assert describe(data={"values": np.arange(4)}, label="a") == ((4,), "a")
    assert describe(label="a", data={"values": np.arange(4)}) == ((4,), "a")
    assert describe(data={"values": np.arange(4).reshape(2, 2)}, label="a") == (
        (2, 2),
        "a",
    )
    assert describe.cache_info()["hits"] == 1


def test_native_cached_result_identity_is_preserved():
    @cache_result()
    def values():
        return np.arange(3)

    result = values()
    assert values() is result


@pytest.mark.parametrize("options", [{"maxsize": 0}, {"ttl": 0}])
def test_disabled_or_immediately_expired_cache_does_not_store(options):
    @cache_result(**options)
    def values():
        return []

    assert values() is not values()
    assert values.cache_info()["size"] == 0
    assert values.cache_info()["misses"] == 2


def test_positive_ttl_expires_using_elapsed_time():
    @cache_result(ttl=0.001)
    def values():
        return []

    first = values()
    time.sleep(0.005)
    assert values() is not first
    assert values.cache_info()["misses"] == 2


def test_unpickleable_arguments_execute_without_unsafe_key_fallback():
    @cache_result()
    def invoke(function):
        return function()

    assert invoke(lambda: 1) == 1
    assert invoke(lambda: 2) == 2
    assert invoke.cache_info()["size"] == 0
    assert invoke.cache_info()["misses"] == 2


def test_failing_custom_serialization_bypasses_cache_without_hiding_interrupts():
    class CannotSerialize:
        def __reduce_ex__(self, protocol):
            raise ValueError("cannot serialize")

    class InterruptedSerialization:
        def __reduce_ex__(self, protocol):
            raise KeyboardInterrupt

    @cache_result()
    def identity(value):
        return value

    value = CannotSerialize()
    assert identity(value) is value
    assert identity.cache_info()["size"] == 0
    with pytest.raises(KeyboardInterrupt):
        identity(InterruptedSerialization())


def test_exceptions_do_not_populate_the_cache():
    @cache_result()
    def fail():
        raise ValueError("native failure")

    for _ in range(2):
        with pytest.raises(ValueError, match="native failure"):
            fail()
    assert fail.cache_info()["size"] == 0
    assert fail.cache_info()["misses"] == 2


@pytest.mark.parametrize("maxsize", [-1, True, 1.5, None])
def test_invalid_capacities_are_rejected(maxsize):
    with pytest.raises(ValueError, match="maxsize"):
        cache_result(maxsize=maxsize)
    with pytest.raises(ValueError, match="maxsize"):
        ComputationCache(maxsize=maxsize)


@pytest.mark.parametrize("ttl", [-1, True, "1", float("nan"), float("inf")])
def test_invalid_lifetimes_are_rejected(ttl):
    with pytest.raises(ValueError, match="ttl"):
        cache_result(ttl=ttl)


def test_cache_updates_preserve_other_entries_and_reads_refresh_lru():
    cache = ComputationCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("b", 3)
    assert cache.info()["size"] == 2
    assert cache.get("a") == 1
    cache.set("c", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 4
    cache.clear()
    assert cache.info() == {"size": 0, "maxsize": 2, "keys": []}


def test_zero_capacity_computation_cache_is_disabled():
    cache = ComputationCache(maxsize=0)
    cache.set("a", 1)
    assert cache.get("a") is None
    assert cache.info()["size"] == 0


def test_parallel_native_computations_preserve_capacity_and_results():
    @cache_result(maxsize=3)
    def square(value):
        return value * value

    cache = ComputationCache(maxsize=3)

    def run(value):
        cache.set(str(value), value)
        cache.get(str(value))
        return square(value)

    values = list(range(10)) * 10
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(run, values)) == [value * value for value in values]
    info = square.cache_info()
    assert info["hits"] + info["misses"] == len(values)
    assert info["size"] <= 3
    assert cache.info()["size"] <= 3
