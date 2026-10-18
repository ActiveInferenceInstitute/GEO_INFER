"""Independent numerical references for communication and entropy contracts."""

import numpy as np
import pytest
from scipy.optimize import minimize

from geo_infer_math.core.information_theory.channel_capacity import (
    awgn_channel_capacity,
    mimo_channel_capacity,
    spatial_channel_capacity,
    waterfilling_power_allocation,
)
from geo_infer_math.core.information_theory.entropy import (
    EntropyCalculator,
    spatial_entropy,
)


@pytest.mark.parametrize("power", [0.0, 0.5, 2.0, 10.0])
def test_mimo_total_power_identity_reference(power):
    expected = 2 * np.log1p(power / 2) / np.log(2)
    assert mimo_channel_capacity(np.eye(2), 1, power) == pytest.approx(expected)


def test_mimo_default_preserves_unit_power_per_transmitter():
    assert mimo_channel_capacity(np.eye(3), 1) == pytest.approx(3)


@pytest.mark.parametrize("power", [None, 0.5, 20.0])
def test_complex_mimo_matches_covariance_log_determinant(power):
    rng = np.random.default_rng(23)
    channel = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    covariance = 1 if power is None else power / channel.shape[1]
    expected = np.linalg.slogdet(
        np.eye(3) + covariance / 0.7 * (channel @ channel.conj().T)
    )[1] / np.log(2)
    actual = mimo_channel_capacity(channel, 0.7, power)
    assert actual == pytest.approx(expected, rel=1e-13)
    left = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
    right = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    assert mimo_channel_capacity(left @ channel @ right, 0.7, power) == pytest.approx(
        actual, rel=1e-13
    )


@pytest.mark.parametrize("magnitude", [1e100, 1e300])
def test_mimo_large_finite_diagonal_never_forms_overflowing_gram_matrix(magnitude):
    with np.errstate(all="raise"):
        actual = mimo_channel_capacity(np.eye(2) * magnitude, 1)
    assert actual == pytest.approx(4 * np.log(magnitude) / np.log(2), rel=1e-14)


def test_mimo_preserves_complex_gain_with_large_finite_components():
    with np.errstate(all="raise"):
        actual = mimo_channel_capacity(np.array([[1e308 + 1e308j]]), 1)
    expected = (2 * np.log(1e308) + np.log(2)) / np.log(2)
    assert actual == pytest.approx(expected, rel=1e-14)


def test_mimo_preserves_small_material_native_mode_at_extreme_dynamic_range():
    channel = np.diag([1e308, 1e-30])
    # The second independent channel contributes about 830 bits at this noise
    # level, even though H/max(abs(H)) would round its amplitude to zero.
    expected = sum(
        np.logaddexp(0, 2 * np.log(gain) - np.log(1e-310)) / np.log(2)
        for gain in [1e308, 1e-30]
    )
    assert mimo_channel_capacity(channel, 1e-310) == pytest.approx(expected, rel=1e-14)


def test_mimo_finite_matrix_with_unrepresentable_singular_values():
    channel = np.array([[1.5e308, 1.5e308], [1.5e308, -1.5e308]])
    with np.errstate(all="raise"):
        actual = mimo_channel_capacity(channel, 1)
    expected = 2 * (2 * np.log(1.5e308) + np.log(2)) / np.log(2)
    assert actual == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize(
    "channel", [[], np.zeros((0, 2)), [1, 2], [[np.nan]], [[np.inf]]]
)
def test_mimo_invalid_channel_rejected(channel):
    with pytest.raises(ValueError):
        mimo_channel_capacity(channel, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_power": 0},
        {"noise_power": -1},
        {"noise_power": np.nan},
        {"noise_power": np.inf},
        {"power_constraint": -1},
        {"power_constraint": np.inf},
        {"base": 1},
        {"base": 0.5},
        {"base": np.nan},
    ],
)
def test_mimo_invalid_physical_parameters_rejected(kwargs):
    options = {"noise_power": 1, **kwargs}
    with pytest.raises(ValueError):
        mimo_channel_capacity(np.eye(2), **options)


def test_waterfilling_known_active_set_and_original_channel_order():
    allocation, capacity = waterfilling_power_allocation([0.25, 4, 1], 1, 2)
    np.testing.assert_allclose(allocation, [0, 1.375, 0.625], atol=1e-14)
    assert capacity == pytest.approx(np.log2(6.5) + np.log2(1.625))


@pytest.mark.parametrize("gains,expected", [([0, 1], [0, 1]), ([0, 0], [0, 0])])
def test_waterfilling_zero_gains_have_zero_allocation_without_warnings(gains, expected):
    with np.errstate(all="raise"):
        allocation, capacity = waterfilling_power_allocation(gains, 1, 1)
    np.testing.assert_array_equal(allocation, expected)
    assert capacity == pytest.approx(sum(expected))


def test_waterfilling_matches_independent_constrained_optimizer():
    gains = np.random.default_rng(42).uniform(0.1, 5, 6)
    noise, power = 0.7, 2.5

    def objective(allocation):
        return -np.log1p(gains * allocation / noise).sum() / np.log(2)

    reference = minimize(
        objective,
        np.full(6, power / 6),
        jac=lambda p: -gains / (noise + gains * p) / np.log(2),
        bounds=[(0, power)] * 6,
        constraints={"type": "eq", "fun": lambda p: p.sum() - power},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 100},
    )
    assert reference.success, reference.message
    allocation, capacity = waterfilling_power_allocation(gains, noise, power)
    np.testing.assert_allclose(allocation, reference.x, atol=2e-6)
    assert allocation.sum() == pytest.approx(power)
    assert capacity == pytest.approx(-reference.fun, abs=1e-10)


@pytest.mark.parametrize(
    "gains,noise,power,expected",
    [
        ([1e-300, 1e300], 1, 1, [0, 1]),
        ([1e-300, 1e-300], 1e300, 2, [1, 1]),
        ([1, 4], 1, 0, [0, 0]),
    ],
)
def test_waterfilling_extreme_scales_and_zero_budget(gains, noise, power, expected):
    with np.errstate(all="raise"):
        allocation, capacity = waterfilling_power_allocation(gains, noise, power)
    np.testing.assert_allclose(allocation, expected, atol=0)
    assert np.isfinite(capacity) and capacity >= 0


@pytest.mark.parametrize(
    "gains,noise,power,base",
    [
        ([-1, 1], 1, 1, 2),
        ([np.nan], 1, 1, 2),
        ([1], 0, 1, 2),
        ([1], 1, -1, 2),
        ([1], 1, np.inf, 2),
        ([1], 1, 1, 1),
    ],
)
def test_waterfilling_invalid_inputs_rejected(gains, noise, power, base):
    with pytest.raises(ValueError):
        waterfilling_power_allocation(gains, noise, power, base)


def test_awgn_resolves_tiny_snr_and_finite_extreme_ratio():
    assert awgn_channel_capacity(1e-20, 1) == pytest.approx(
        np.log1p(1e-20) / np.log(2), rel=1e-14, abs=0
    )
    with np.errstate(all="raise"):
        actual = awgn_channel_capacity(1e300, 1e-300)
    assert actual == pytest.approx(600 * np.log(10) / np.log(2))
    assert awgn_channel_capacity(0, 1) == 0
    assert awgn_channel_capacity(1, 1, bandwidth=0) == 0


@pytest.mark.parametrize(
    "signal,noise,bandwidth,base",
    [
        (-1, 1, 1, 2),
        (1, 0, 1, 2),
        (1, 1, -1, 2),
        (np.nan, 1, 1, 2),
        (1, np.inf, 1, 2),
        (1, 1, 1, 1),
    ],
)
def test_awgn_invalid_physical_parameters_rejected(signal, noise, bandwidth, base):
    with pytest.raises(ValueError):
        awgn_channel_capacity(signal, noise, bandwidth, base)


def test_spatial_channel_zero_and_small_signal_have_physical_capacity():
    with np.errstate(all="raise"):
        assert spatial_channel_capacity([[1, 0]], [0], 1) == 0
        capacity = spatial_channel_capacity([[1, 0]], [1e-20], 1)
    assert capacity == pytest.approx(np.log1p(1e-20) / np.log(2), rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "coordinates,signal,kwargs",
    [
        ([1, 0], [1], {}),
        ([[1, 0]], [1, 2], {}),
        ([[1, 0]], [-1], {}),
        ([[np.inf, 0]], [1], {}),
        ([[1, 0]], [1], {"noise_power": 0}),
        ([[1, 0]], [1], {"path_loss_exponent": 0}),
        ([[1, 0]], [1], {"base": 1}),
    ],
)
def test_spatial_channel_invalid_physical_inputs_rejected(coordinates, signal, kwargs):
    options = {"noise_power": 1, **kwargs}
    with pytest.raises(ValueError):
        spatial_channel_capacity(coordinates, signal, **options)


@pytest.mark.parametrize("base", [2, np.e, 10])
@pytest.mark.parametrize("method", ["shannon", "renyi"])
@pytest.mark.parametrize("use_values", [False, True])
def test_spatial_entropy_and_calculator_preserve_requested_units(
    base, method, use_values
):
    coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.arange(4) if use_values else None
    bins = 4 if use_values else 2
    expected = np.log(4) / np.log(base)
    assert spatial_entropy(
        coordinates, values, method, bins=bins, base=base
    ) == pytest.approx(expected)
    assert EntropyCalculator(base).spatial_entropy(
        coordinates, values, method, bins=bins
    ) == pytest.approx(expected)
