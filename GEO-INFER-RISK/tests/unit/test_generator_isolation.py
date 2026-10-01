"""Tests that every stochastic RISK component draws from its own generator.

The property under test is isolation, not just replay. Global-state randomness
fails in a way that is easy to miss: two runs of the same code agree as long as
nothing else in the process draws in between, then diverge the moment an
unrelated caller does. These tests pin the seed contract for each component and
assert that the process-wide ``numpy.random`` stream is neither read nor
advanced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_infer_risk.core.catastrophe_models import (
    CatastropheConfig,
    EnhancedCatastropheModel,
    EnhancedEarthquakeModel,
    EnhancedFloodModel,
    EnhancedHurricaneModel,
)
from geo_infer_risk.core.exposure_model import EnhancedExposureModel
from geo_infer_risk.core.hazard_model import EnhancedHazardModel
from geo_infer_risk.core.risk_models import (
    BuildingVulnerabilityModel,
    FloodHazardModel,
    PopulationExposureModel,
    RiskParameters,
)
from geo_infer_risk.core.vulnerability_model import EnhancedVulnerabilityModel


def global_stream_untouched(action: object) -> bool:
    """Return whether ``action`` left the numpy.random singleton alone.

    Args:
        action: Zero-argument callable to run.

    Returns:
        True if the next draw from the global stream is what it would have been
        had ``action`` never run.
    """
    np.random.seed(4242)
    expected = np.random.random()
    np.random.seed(4242)
    action()  # type: ignore[operator]
    return bool(np.random.random() == expected)


def earthquake_model(**kwargs: object) -> EnhancedEarthquakeModel:
    """A minimal earthquake model whose hazard generation runs."""
    config = CatastropheConfig(
        simulation_years=10,
        return_periods=[10, 25],
        spatial_correlation=False,
        batch_size=10,
        **kwargs,  # type: ignore[arg-type]
    )
    model = EnhancedEarthquakeModel(config=config)
    model.model_parameters = {"mean_depth": 15.0}
    return model


class TestCatastropheModelSeeding:
    def test_config_seed_makes_construction_replayable(self) -> None:
        a = earthquake_model(random_seed=3).simulate_events(6)
        b = earthquake_model(random_seed=3).simulate_events(6)
        assert [e["event_id"] for e in a] == [e["event_id"] for e in b]

    def test_constructor_seed_overrides_the_config_seed(self) -> None:
        config = CatastropheConfig(spatial_correlation=False, random_seed=1)
        first = EnhancedEarthquakeModel(config=config, random_seed=99)
        second = EnhancedEarthquakeModel(config=config, random_seed=99)
        for model in (first, second):
            model.model_parameters = {"mean_depth": 15.0}
        assert [e["event_id"] for e in first.simulate_events(4)] == [
            e["event_id"] for e in second.simulate_events(4)
        ]

    def test_successive_calls_draw_fresh_events(self) -> None:
        """A seed set once must not make every later call repeat itself."""
        model = earthquake_model(random_seed=5)
        first = [e["event_id"] for e in model.simulate_events(4)]
        second = [e["event_id"] for e in model.simulate_events(4)]
        assert first != second

    def test_leaves_global_stream_untouched(self) -> None:
        assert global_stream_untouched(lambda: earthquake_model().simulate_events(6))

    @pytest.mark.parametrize(
        "model_class",
        [EnhancedEarthquakeModel, EnhancedHurricaneModel, EnhancedFloodModel],
    )
    def test_every_peril_model_forwards_its_seed(self, model_class: type) -> None:
        config = CatastropheConfig(spatial_correlation=False)
        a = model_class(config=config, random_seed=8)
        b = model_class(config=config, random_seed=8)
        assert a._rng.random() == b._rng.random()

    def test_generator_is_never_the_numpy_random_module(self) -> None:
        model = EnhancedCatastropheModel()
        assert isinstance(model._rng, np.random.Generator)

    def test_zero_width_time_window_returns_its_start(self) -> None:
        """A degenerate window must not raise from integers(0, 0)."""
        from datetime import datetime

        model = earthquake_model(random_seed=2)
        moment = datetime(2026, 1, 1)
        assert model._generate_event_timestamp((moment, moment)) == moment


class TestExposureModelSeeding:
    def test_zero_is_a_valid_seed(self) -> None:
        """A falsy seed must not be silently discarded."""
        a = EnhancedExposureModel("property", {"random_seed": 0})
        b = EnhancedExposureModel("property", {"random_seed": 0})
        assert a._rng.random() == b._rng.random()

    def test_removed_seed_alias_is_rejected(self) -> None:
        """Only ``random_seed`` seeds the model; a ``seed`` key fails loudly."""
        with pytest.raises(ValueError, match="random_seed"):
            EnhancedExposureModel("property", {"seed": 12})
        with pytest.raises(ValueError, match="random_seed"):
            EnhancedExposureModel("property", {"random_seed": 1, "seed": 2})

    def test_unseeded_models_are_independent(self) -> None:
        a = EnhancedExposureModel("property", {})
        b = EnhancedExposureModel("property", {})
        assert a._rng.random() != b._rng.random()

    def test_generator_is_never_the_numpy_random_module(self) -> None:
        model = EnhancedExposureModel("property", {})
        assert isinstance(model._rng, np.random.Generator)


class TestHazardAndVulnerabilitySeeding:
    def test_hazard_model_replays(self) -> None:
        a = EnhancedHazardModel("flood", {"random_seed": 6})
        b = EnhancedHazardModel("flood", {"random_seed": 6})
        assert a.rng.random() == b.rng.random()

    def test_hazard_model_accepts_a_generator(self) -> None:
        a = EnhancedHazardModel("flood", {"random_seed": np.random.default_rng(6)})
        b = EnhancedHazardModel("flood", {"random_seed": np.random.default_rng(6)})
        assert a.rng.random() == b.rng.random()

    def test_hazard_model_rejects_an_unusable_seed(self) -> None:
        with pytest.raises(TypeError, match="seed must be"):
            EnhancedHazardModel("flood", {"random_seed": "not-a-seed"})

    def test_vulnerability_uncertainty_leaves_global_stream_untouched(self) -> None:
        model = EnhancedVulnerabilityModel("building", {})
        assert global_stream_untouched(lambda: model._apply_uncertainty(0.4))

    def test_vulnerability_zero_seed_replays(self) -> None:
        a = EnhancedVulnerabilityModel("building", {"random_seed": 0})
        b = EnhancedVulnerabilityModel("building", {"random_seed": 0})
        assert a._apply_uncertainty(0.4) == b._apply_uncertainty(0.4)


class TestRiskModelSampling:
    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: FloodHazardModel(),
            lambda: BuildingVulnerabilityModel(),
            lambda: PopulationExposureModel(),
        ],
    )
    def test_component_sample_accepts_an_int_seed(self, model_factory: object) -> None:
        """random_state used to be `or`-ed, so 0 fell through to a fresh RNG."""
        first = model_factory()  # type: ignore[operator]
        second = model_factory()  # type: ignore[operator]
        np.testing.assert_array_equal(
            first.sample(random_state=0), second.sample(random_state=0)
        )

    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: FloodHazardModel(),
            lambda: BuildingVulnerabilityModel(),
            lambda: PopulationExposureModel(),
        ],
    )
    def test_component_sample_leaves_global_stream_untouched(
        self, model_factory: object
    ) -> None:
        model = model_factory()  # type: ignore[operator]
        assert global_stream_untouched(model.sample)

    def test_risk_parameters_reject_an_unusable_seed(self) -> None:
        with pytest.raises(TypeError, match="seed must be"):
            RiskParameters(random_seed="not-a-seed")  # type: ignore[arg-type]

    def test_risk_parameters_accept_zero_and_a_generator(self) -> None:
        assert RiskParameters(random_seed=0).random_seed == 0
        generator = np.random.default_rng(1)
        assert RiskParameters(random_seed=generator).random_seed is generator


class TestEndToEndReplay:
    def test_a_seeded_loss_table_replays_exactly(self) -> None:
        """The property a risk number needs: rerun the run, get the run back."""

        def run() -> pd.DataFrame:
            model = earthquake_model(random_seed=17)
            events = model.simulate_events(25)
            return pd.DataFrame(
                {
                    "event_id": [e["event_id"] for e in events],
                    "hazard_type": ["earthquake"] * len(events),
                    "loss": [float(e.get("magnitude", 0.0)) for e in events],
                }
            )

        pd.testing.assert_frame_equal(run(), run())
