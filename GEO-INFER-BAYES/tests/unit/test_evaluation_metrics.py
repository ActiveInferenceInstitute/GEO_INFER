"""
Unit tests for the probabilistic model-evaluation metrics in core/evaluation.py.

Covers the strictly proper scoring rules (CRPS, log predictive density), the
quantile pinball loss, and the calibration diagnostics (empirical coverage,
interval score, PIT uniformity) that judge a predictive *distribution* rather
than a point forecast.
"""

import numpy as np
import pytest


from geo_infer_bayes.core.evaluation import (
    coverage_calibration_error,
    crps,
    crps_gaussian,
    crps_pointwise,
    empirical_coverage,
    evaluate_gaussian,
    evaluate_predictive,
    interval_score,
    log_predictive_density,
    log_predictive_density_gaussian,
    pinball_loss,
    pit_gaussian,
    pit_uniformity_statistic,
    pit_values,
)


class TestCrps:
    def test_sample_matches_gaussian_closed_form(self) -> None:
        """As draws grow the empirical CRPS must converge to the analytic value."""
        rng = np.random.default_rng(0)
        obs = np.array([0.5, 1.2, -0.3, 2.0])
        mean = np.zeros(4)
        std = np.ones(4)
        draws = rng.normal(mean, std, size=(200_000, 4))
        assert crps(obs, draws) == pytest.approx(
            crps_gaussian(obs, mean, std), abs=0.005
        )

    def test_sharp_centered_predictive_scores_low(self) -> None:
        """A predictive pinned to the observations scores near zero."""
        obs = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(1)
        draws = rng.normal(obs, 1e-6, size=(500, 3))
        assert crps(obs, draws) < 0.05
        assert crps_gaussian(obs, obs, np.full(3, 1e-6)) < 0.05

    def test_diffuse_predictive_scores_higher(self) -> None:
        """A wide predictive around the same mean scores worse than a tight one."""
        obs = np.array([0.0, 0.0])
        tight = np.random.default_rng(2).normal(0.0, 0.1, size=(4000, 2))
        wide = np.random.default_rng(2).normal(0.0, 3.0, size=(4000, 2))
        assert crps(obs, wide) > crps(obs, tight)

    def test_pointwise_shape(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, size=(500, 3))
        per_point = crps_pointwise(np.zeros(3), draws)
        assert per_point.shape == (3,)
        assert np.all(per_point >= 0.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="must match observations"):
            crps(np.zeros(3), rng.normal(size=(5, 4)))

    def test_non_finite_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            crps(np.array([0.0, np.nan]), np.zeros((3, 2)))


class TestPinballLoss:
    def test_median_reduces_to_half_the_mae(self) -> None:
        """At quantile 0.5 the pinball loss is exactly 0.5 times the MAE."""
        obs = np.array([1.0, -2.0, 0.5, 3.0])
        pred = np.array([0.9, -1.8, 0.4, 2.9])
        assert pinball_loss(obs, pred, 0.5) == pytest.approx(
            0.5 * np.mean(np.abs(obs - pred))
        )

    def test_lower_quantile_punishes_underruns(self) -> None:
        """A high forecast at a low quantile level should score poorly."""
        obs = np.array([1.0, 1.0, 1.0])
        high = np.full(3, 5.0)
        low = np.full(3, -5.0)
        assert pinball_loss(obs, high, 0.1) > pinball_loss(obs, low, 0.1)

    def test_level_outside_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly"):
            pinball_loss(np.zeros(3), np.zeros(3), 0.0)


class TestCoverageDiagnostics:
    def test_empirical_coverage_bounds(self) -> None:
        obs = np.array([0.0, 2.0, 4.0])
        assert empirical_coverage(obs, -1.0, 5.0) == pytest.approx(1.0)
        assert empirical_coverage(obs, 10.0, 20.0) == pytest.approx(0.0)

    def test_coverage_calibration_error(self) -> None:
        obs = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        # [1, 3, 5, 7, 9] contains observations 2,4,6,8 -> 4 / 5 = 0.8.
        assert coverage_calibration_error(
            obs, np.ones(5), np.full(5, 9.0), 0.95
        ) == pytest.approx(abs(0.8 - 0.95))

    def test_interval_score_rewards_narrow_intervals(self) -> None:
        obs = np.array([0.0, 0.0, 0.0])
        narrow = interval_score(obs, -0.5, 0.5, 0.95)
        wide = interval_score(obs, -5.0, 5.0, 0.95)
        assert narrow < wide

    def test_interval_misses_pay_a_penalty(self) -> None:
        obs = np.array([0.0])
        covered = interval_score(obs, -1.0, 1.0, 0.95)
        missed = interval_score(obs, 2.0, 3.0, 0.95)  # observation below interval
        assert missed > covered

    def test_inverted_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="upper"):
            empirical_coverage(np.zeros(3), 1.0, -1.0)


class TestPitCalibration:
    def test_gaussian_pit_is_uniform(self) -> None:
        """Calibrated predictions make PIT values look uniform on [0, 1]."""
        rng = np.random.default_rng(0)
        n = 2000
        draws = rng.normal(0.0, 1.0, size=(500, n))
        obs = rng.normal(0.0, 1.0, size=n)
        pits = pit_values(obs, draws)
        assert abs(float(pits.mean()) - 0.5) < 0.05
        assert np.all(pits >= 0.0) and np.all(pits <= 1.0)
        assert pit_uniformity_statistic(obs, draws, n_bins=50) < 0.2

    def test_pit_gaussian_tracks_the_observables(self) -> None:
        obs = np.array([0.0, 1.0])
        pits = pit_gaussian(obs, np.zeros(2), np.ones(2))
        assert pits[0] == pytest.approx(0.5, abs=1e-9)
        assert pits[1] == pytest.approx(0.8413, abs=1e-3)


class TestLogPredictiveDensity:
    def test_centered_predictive_assigns_more_probability(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, size=(600, 1))
        center = log_predictive_density(np.array([0.0]), draws)
        far = log_predictive_density(np.array([6.0]), draws)
        assert center > far

    def test_gaussian_closed_form(self) -> None:
        obs = np.array([0.0, 1.0])
        assert log_predictive_density_gaussian(
            obs, np.zeros(2), np.ones(2)
        ) == pytest.approx(-0.5 * np.log(2 * np.pi) - 0.5 * 0.5, abs=1e-9)


class TestEvaluateConvenience:
    def test_evaluate_predictive_keys(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, size=(300, 10))
        obs = rng.normal(0.0, 1.0, size=10)
        report = evaluate_predictive(obs, draws, level=0.95)
        assert set(report) == {
            "crps",
            "mean_absolute_error",
            "log_predictive_density",
            "coverage",
            "coverage_deviation",
            "interval_score",
            "pit_uniformity",
        }
        assert 0.6 < report["coverage"] < 1.0

    def test_evaluate_gaussian_keys(self) -> None:
        rng = np.random.default_rng(1)
        obs = rng.normal(0.0, 1.0, size=50)
        report = evaluate_gaussian(obs, np.zeros(50), np.ones(50))
        assert set(report) == {
            "crps",
            "mean_absolute_error",
            "log_predictive_density",
            "coverage",
            "coverage_deviation",
            "interval_score",
            "pit_uniformity",
        }
        assert report["coverage"] == pytest.approx(0.95, abs=0.15)

    def test_invalid_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly between zero and one"):
            evaluate_predictive(np.zeros(2), np.zeros((3, 2)), level=1.0)
