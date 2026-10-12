"""Tests for the catastrophe risk metrics in ``geo_infer_risk.utils.risk_metrics``.

The exceedance-probability family is checked against closed-form answers rather
than against itself: for a table of ``n`` distinct events the Weibull plotting
position puts the ``i``-th largest loss at exceedance probability
``i / (n + 1)``, so the loss at any requested probability is a linear
interpolation between known points. That is what makes a regression such as
reading the curve backwards detectable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from geo_infer_risk.utils.risk_metrics import (
    calculate_aal,
    calculate_annual_aggregate_exceedance_probability,
    calculate_annual_occurrence_exceedance_probability,
    calculate_correlation_matrix,
    calculate_ep_curve,
    calculate_loss_by_return_period,
    calculate_loss_frequency_curve,
    calculate_pml,
    calculate_tail_value_at_risk,
)


@pytest.fixture
def ramp_table() -> pd.DataFrame:
    """Ten events with losses 100, 200, ... 1000."""
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(10)],
            "hazard_type": ["eq"] * 10,
            "loss": [100.0 * (i + 1) for i in range(10)],
        }
    )


@pytest.fixture
def empty_table() -> pd.DataFrame:
    return pd.DataFrame({"event_id": [], "hazard_type": [], "loss": []})


def weibull_reference(losses: list[float], prob: float) -> float:
    """Closed-form loss at an exceedance probability, computed independently."""
    ordered = sorted(losses, reverse=True)
    positions = [(i + 1) / (len(ordered) + 1) for i in range(len(ordered))]
    return float(np.interp(prob, positions, ordered))


class TestAal:
    def test_array_input_one_event_per_year_is_mean_event_loss(self) -> None:
        losses = np.array([1000.0, 2000.0, 500.0, 3000.0, 1500.0])
        assert calculate_aal(losses, exposure_years=5.0) == pytest.approx(1600.0)

    def test_array_input_annualizes_when_given_exposure_years(self) -> None:
        losses = np.array([100.0, 300.0])
        assert calculate_aal(losses, exposure_years=4.0) == pytest.approx(100.0)

    def test_empty_array_is_zero(self) -> None:
        assert calculate_aal(np.array([]), exposure_years=1.0) == 0.0

    def test_rejects_two_dimensional_array(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            calculate_aal(np.zeros((2, 2)), exposure_years=1.0)

    def test_dataframe_splits_by_hazard(self) -> None:
        table = pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "hazard_type": ["flood", "flood", "wind"],
                "loss": [100.0, 300.0, 200.0],
            }
        )
        result = calculate_aal(table, exposure_years=2.0)
        assert result["total"] == pytest.approx(300.0)
        assert result["by_hazard"] == {
            "flood": pytest.approx(200.0),
            "wind": pytest.approx(100.0),
        }

    def test_missing_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hazard_type"):
            calculate_aal(
                pd.DataFrame({"event_id": ["a"], "loss": [1.0]}), exposure_years=1.0
            )


class TestEpCurve:
    @pytest.mark.parametrize("prob", [0.5, 0.2, 0.1, 0.05, 1.0])
    def test_matches_closed_form_interpolation(
        self, ramp_table: pd.DataFrame, prob: float
    ) -> None:
        curve = calculate_ep_curve(ramp_table, [prob])
        expected = weibull_reference(list(ramp_table["loss"]), prob)
        assert curve["loss"][0] == pytest.approx(expected)

    def test_median_exceedance_is_a_middling_loss_not_the_smallest(
        self, ramp_table: pd.DataFrame
    ) -> None:
        """Regression guard: reading the curve backwards returned the minimum."""
        loss = calculate_ep_curve(ramp_table, [0.5])["loss"][0]
        assert 400.0 < loss < 700.0

    def test_rarer_probabilities_give_larger_losses(
        self, ramp_table: pd.DataFrame
    ) -> None:
        curve = calculate_ep_curve(ramp_table, [0.5, 0.2, 0.1])
        assert curve["exceedance_probability"] == [0.5, 0.2, 0.1]
        assert curve["loss"] == sorted(curve["loss"])

    def test_full_curve_is_returned_when_no_probabilities_requested(
        self, ramp_table: pd.DataFrame
    ) -> None:
        curve = calculate_ep_curve(ramp_table)
        assert len(curve["loss"]) == 10
        assert curve["loss"] == sorted(curve["loss"], reverse=True)
        assert curve["exceedance_probability"][0] == pytest.approx(1 / 11)
        assert curve["return_period"][0] == pytest.approx(11.0)

    def test_multi_row_events_are_summed_once(self) -> None:
        """Two rows for one event are one event with the summed loss."""
        table = pd.DataFrame(
            {
                "event_id": ["a", "a", "b"],
                "hazard_type": ["wind", "surge", "wind"],
                "loss": [40.0, 60.0, 10.0],
            }
        )
        curve = calculate_ep_curve(table)
        assert curve["loss"] == [100.0, 10.0]

    def test_exposure_years_converts_to_annual_probability(
        self, ramp_table: pd.DataFrame
    ) -> None:
        """With 10 events over 10 years, annual rates are Poisson-mapped."""
        curve = calculate_ep_curve(ramp_table, exposure_years=10.0)
        # The largest loss is exceeded once in 10 years -> rate 1/10.
        assert curve["exceedance_probability"][0] == pytest.approx(
            1.0 - np.exp(-1 / 11)
        )
        assert all(0.0 < p < 1.0 for p in curve["exceedance_probability"])

    def test_annual_probabilities_are_lower_than_per_event(
        self, ramp_table: pd.DataFrame
    ) -> None:
        """A 10-year record spreads the same events over more time."""
        per_event = calculate_ep_curve(ramp_table)["exceedance_probability"]
        annual = calculate_ep_curve(ramp_table, exposure_years=10.0)[
            "exceedance_probability"
        ]
        assert all(a < e for a, e in zip(annual, per_event))

    @pytest.mark.parametrize("prob", [0.0, -0.1, 1.5])
    def test_out_of_range_probability_is_rejected(
        self, ramp_table: pd.DataFrame, prob: float
    ) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            calculate_ep_curve(ramp_table, [prob])

    def test_empty_table_gives_empty_curve(self, empty_table: pd.DataFrame) -> None:
        curve = calculate_ep_curve(empty_table)
        assert curve == {
            "exceedance_probability": [],
            "loss": [],
            "return_period": [],
        }


class TestPml:
    def test_matches_the_ep_curve_at_the_same_probability(
        self, ramp_table: pd.DataFrame
    ) -> None:
        assert calculate_pml(ramp_table, return_period=4) == pytest.approx(
            calculate_ep_curve(ramp_table, [0.25])["loss"][0]
        )

    def test_warns_when_the_return_period_exceeds_the_record(
        self, ramp_table: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="geo_infer_risk"):
            pml = calculate_pml(ramp_table, return_period=250)
        assert pml == pytest.approx(1000.0)  # clamped to the largest loss
        assert any("understates the tail" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("return_period", [0.5, 1.0, -3.0, float("inf")])
    def test_rejects_return_periods_of_one_year_or_less(
        self, ramp_table: pd.DataFrame, return_period: float
    ) -> None:
        with pytest.raises(ValueError, match="return_period"):
            calculate_pml(ramp_table, return_period=return_period)

    def test_empty_table_is_zero(self, empty_table: pd.DataFrame) -> None:
        assert calculate_pml(empty_table) == 0.0


class TestLossByReturnPeriod:
    def test_preserves_the_requested_order(self, ramp_table: pd.DataFrame) -> None:
        result = calculate_loss_by_return_period(ramp_table, [10, 2, 5])
        assert list(result) == ["10.0", "2.0", "5.0"]
        assert result["10.0"] > result["5.0"] > result["2.0"]

    def test_agrees_with_pml_per_period(self, ramp_table: pd.DataFrame) -> None:
        result = calculate_loss_by_return_period(ramp_table, [5])
        assert result["5.0"] == pytest.approx(calculate_pml(ramp_table, 5))

    def test_empty_request_gives_empty_result(self, ramp_table: pd.DataFrame) -> None:
        assert calculate_loss_by_return_period(ramp_table, []) == {}

    def test_rejects_a_bad_period(self, ramp_table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="return period"):
            calculate_loss_by_return_period(ramp_table, [10, 0.5])


class TestTailValueAtRisk:
    def test_is_the_mean_of_the_breaching_tail(self, ramp_table: pd.DataFrame) -> None:
        # ceil(0.9 * 10) - 1 = 8 -> VaR is the 9th smallest loss, 900.
        assert calculate_tail_value_at_risk(ramp_table, 0.9) == pytest.approx(950.0)

    def test_never_below_var(self, ramp_table: pd.DataFrame) -> None:
        for level in (0.5, 0.75, 0.95, 0.99):
            tvar = calculate_tail_value_at_risk(ramp_table, level)
            assert tvar >= np.quantile(ramp_table["loss"], level, method="higher")

    def test_increases_with_confidence_level(self, ramp_table: pd.DataFrame) -> None:
        levels = [0.5, 0.7, 0.9, 0.99]
        values = [calculate_tail_value_at_risk(ramp_table, x) for x in levels]
        assert values == sorted(values)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.2, float("nan")])
    def test_rejects_levels_outside_the_open_unit_interval(
        self, ramp_table: pd.DataFrame, level: float
    ) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            calculate_tail_value_at_risk(ramp_table, level)

    def test_empty_table_is_zero(self, empty_table: pd.DataFrame) -> None:
        assert calculate_tail_value_at_risk(empty_table) == 0.0


class TestOccurrenceExceedanceProbability:
    def test_matches_the_poisson_closed_form(self, ramp_table: pd.DataFrame) -> None:
        # Losses 900 and 1000 exceed 850, over 5 years -> rate 0.4.
        oep = calculate_annual_occurrence_exceedance_probability(
            ramp_table, threshold=850.0, exposure_years=5.0
        )
        assert oep == pytest.approx(1.0 - np.exp(-0.4))

    def test_longer_records_give_lower_annual_probability(
        self, ramp_table: pd.DataFrame
    ) -> None:
        short = calculate_annual_occurrence_exceedance_probability(
            ramp_table, 850.0, exposure_years=2.0
        )
        long = calculate_annual_occurrence_exceedance_probability(
            ramp_table, 850.0, exposure_years=20.0
        )
        assert long < short

    def test_threshold_above_every_loss_is_zero(self, ramp_table: pd.DataFrame) -> None:
        assert (
            calculate_annual_occurrence_exceedance_probability(
                ramp_table, 10_000.0, exposure_years=1.0
            )
            == 0.0
        )

    def test_warns_when_exposure_years_is_omitted(
        self, ramp_table: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="geo_infer_risk"):
            calculate_annual_occurrence_exceedance_probability(ramp_table, 850.0)
        assert any("exposure_years" in rec.message for rec in caplog.records)

    def test_rejects_a_non_finite_threshold(self, ramp_table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="threshold"):
            calculate_annual_occurrence_exceedance_probability(ramp_table, float("nan"))

    @pytest.mark.parametrize("years", [0.0, -1.0, float("inf")])
    def test_rejects_non_positive_exposure_years(
        self, ramp_table: pd.DataFrame, years: float
    ) -> None:
        with pytest.raises(ValueError, match="exposure_years"):
            calculate_annual_occurrence_exceedance_probability(
                ramp_table, 850.0, exposure_years=years
            )


class TestAggregateExceedanceProbability:
    def test_threshold_of_zero_is_almost_certainly_exceeded(
        self, ramp_table: pd.DataFrame
    ) -> None:
        aep = calculate_annual_aggregate_exceedance_probability(
            ramp_table, threshold=0.0, num_years=2000, random_seed=1, exposure_years=1.0
        )
        # Only a year that draws zero events fails to exceed 0; at rate 10 that
        # is exp(-10), far below the Monte Carlo resolution here.
        assert aep > 0.99

    def test_unreachable_threshold_is_zero(self, ramp_table: pd.DataFrame) -> None:
        aep = calculate_annual_aggregate_exceedance_probability(
            ramp_table,
            threshold=1e12,
            num_years=500,
            random_seed=1,
            exposure_years=1.0,
        )
        assert aep == 0.0

    def test_estimate_is_within_monte_carlo_error_of_a_long_run(
        self, ramp_table: pd.DataFrame
    ) -> None:
        """Two independent seeds agree to within a few standard errors."""
        kwargs = {"threshold": 5500.0, "num_years": 40_000, "exposure_years": 1.0}
        a = calculate_annual_aggregate_exceedance_probability(
            ramp_table, random_seed=11, **kwargs
        )
        b = calculate_annual_aggregate_exceedance_probability(
            ramp_table, random_seed=22, **kwargs
        )
        standard_error = np.sqrt(max(a, 1e-4) * (1 - a) / 40_000)
        assert abs(a - b) < 5 * standard_error

    def test_longer_exposure_lowers_the_annual_rate_and_the_probability(
        self, ramp_table: pd.DataFrame
    ) -> None:
        dense = calculate_annual_aggregate_exceedance_probability(
            ramp_table, 3000.0, num_years=5000, random_seed=3, exposure_years=1.0
        )
        sparse = calculate_annual_aggregate_exceedance_probability(
            ramp_table, 3000.0, num_years=5000, random_seed=3, exposure_years=10.0
        )
        assert sparse < dense

    def test_empty_table_is_zero(self, empty_table: pd.DataFrame) -> None:
        assert (
            calculate_annual_aggregate_exceedance_probability(
                empty_table, threshold=1.0, num_years=10
            )
            == 0.0
        )

    @pytest.mark.parametrize("num_years", [0, -5, 2.5])
    def test_rejects_a_bad_year_count(
        self, ramp_table: pd.DataFrame, num_years: object
    ) -> None:
        with pytest.raises(ValueError, match="num_years"):
            calculate_annual_aggregate_exceedance_probability(
                ramp_table,
                threshold=1.0,
                num_years=num_years,  # type: ignore[arg-type]
            )


class TestLossFrequencyCurve:
    def test_bins_and_normalization(self, ramp_table: pd.DataFrame) -> None:
        curve = calculate_loss_frequency_curve(ramp_table, num_bins=5)
        assert len(curve["bin_edges"]) == 6
        assert sum(curve["frequencies"]) == 10
        assert sum(curve["normalized_frequencies"]) == pytest.approx(1.0)

    def test_empty_table_gives_empty_lists(self, empty_table: pd.DataFrame) -> None:
        assert calculate_loss_frequency_curve(empty_table) == {
            "bin_edges": [],
            "frequencies": [],
            "normalized_frequencies": [],
        }

    @pytest.mark.parametrize("num_bins", [0, -1, 2.5])
    def test_rejects_a_bad_bin_count(
        self, ramp_table: pd.DataFrame, num_bins: object
    ) -> None:
        with pytest.raises(ValueError, match="num_bins"):
            calculate_loss_frequency_curve(ramp_table, num_bins=num_bins)  # type: ignore[arg-type]


class TestCorrelationMatrix:
    def test_perfectly_correlated_hazards(self) -> None:
        table = pd.DataFrame(
            {
                "event_id": ["a", "a", "b", "b", "c", "c"],
                "hazard_type": ["wind", "surge"] * 3,
                "loss": [10.0, 20.0, 20.0, 40.0, 30.0, 60.0],
            }
        )
        result = calculate_correlation_matrix(table)
        assert result["hazard_types"] == ["surge", "wind"]
        assert result["correlation_matrix"][0][1] == pytest.approx(1.0)

    def test_missing_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="event_id"):
            calculate_correlation_matrix(
                pd.DataFrame({"hazard_type": ["wind"], "loss": [1.0]})
            )
