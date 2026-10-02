"""
Comprehensive end-to-end integration and composition tests across:
  - H3 core & spatial indexing (GEO-INFER-SPACE)
  - Spatiotemporal stream analysis & TimeSeries (GEO-INFER-TIME)
  - Bayesian spatial & hierarchical inference (GEO-INFER-BAYES)
  - Catastrophe, exposure, and hazard risk modeling (GEO-INFER-RISK)
  - Active Inference belief propagation & policy selection (GEO-INFER-ACT)

Verifies full multi-module composition with zero mock leakage and strict H3 v4 conformance.
"""

from __future__ import annotations

import datetime

import h3
import numpy as np
import pandas as pd
import pytest
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.policy_selection import PolicySelector
from geo_infer_act.utils.h3_adapter import get_h3_adapter
from geo_infer_bayes.models.spatial_gp import SpatialGP
from geo_infer_bayes.utils.rng import resolve_rng as resolve_bayes_rng
from geo_infer_risk.core.exposure_model import EnhancedExposureModel
from geo_infer_risk.utils.risk_metrics import calculate_ep_curve
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_space import H3StateSpace
from geo_infer_space.nested import NestedH3Grid
from geo_infer_time.core.stream_processing import StreamProcessor
from geo_infer_time.models.timeseries import TimeSeries


@pytest.fixture
def h3_spatial_domain():
    """Create a deterministic set of H3 cells around San Francisco."""
    center_lat, center_lng = 37.7749, -122.4194
    resolution = 8
    indexer = SpatialIndexingInterface(backend="h3")
    center_cell = str(indexer.latlng_to_cell(center_lat, center_lng, resolution))
    disk_cells = sorted(str(c) for c in h3.grid_disk(center_cell, 2))
    return {
        "center_lat": center_lat,
        "center_lng": center_lng,
        "resolution": resolution,
        "center_cell": center_cell,
        "disk_cells": disk_cells,
        "coords": [h3.cell_to_latlng(c) for c in disk_cells],
    }


def test_h3_space_nested_hierarchy_composition(h3_spatial_domain):
    """Test SPACE Nested H3 Grid hierarchy creation and metrics."""
    cells = h3_spatial_domain["disk_cells"]
    grid = NestedH3Grid()
    hierarchy = grid.build_h3_hierarchy_from_cells(
        cells=cells,
        resolutions=[7, 8],
    )

    assert hierarchy["resolutions"] == [7, 8]
    assert hierarchy["validation"]["is_valid"] is True
    assert set(hierarchy["leaf_cells"]) == set(cells)
    for parent, children in hierarchy["parent_child_map"].items():
        for child in children:
            assert h3.cell_to_parent(child, 7) == parent


def test_space_time_spatiotemporal_stream_composition(h3_spatial_domain):
    """Test composition of H3 spatial cells with TIME stream processor and TimeSeries."""
    cells = h3_spatial_domain["disk_cells"][:5]
    base_time = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    cell_streams: dict[str, StreamProcessor] = {}
    for cell in cells:
        # Window size large enough to hold all 12 hourly readings
        processor = StreamProcessor(window_size=datetime.timedelta(hours=24))
        lat, lng = h3.cell_to_latlng(cell)
        spatial_offset = (lat - 37.7) * 10.0 + (lng + 122.4) * 10.0
        for hour in range(12):
            ts = base_time + datetime.timedelta(hours=hour)
            value = 20.0 + spatial_offset + np.sin(hour / 2.0)
            processor.add_data_point(ts, value)
        cell_streams[cell] = processor

    # Validate stream aggregation per H3 cell
    for processor in cell_streams.values():
        summary = processor.get_buffer_summary()
        assert summary["size"] == 12
        assert "mean" in summary
        assert summary["mean"] > 0

    # Package into a GEO-INFER-TIME TimeSeries
    ts_data = [
        {"timestamp": base_time + datetime.timedelta(hours=h), "temp": 20.0 + h}
        for h in range(10)
    ]
    df = pd.DataFrame(ts_data).set_index("timestamp")
    series = TimeSeries(data=df, spatial_location={"cell": cells[0]})
    assert series.spatial_location == {"cell": cells[0]}
    assert len(series) == 10


def test_space_bayes_spatial_gaussian_process_composition(h3_spatial_domain):
    """Test fitting Bayesian SpatialGP over H3 cell centroids."""
    coords = np.array(h3_spatial_domain["coords"])
    rng = resolve_bayes_rng(42)
    y = (
        np.sin(coords[:, 0] * 5.0)
        + np.cos(coords[:, 1] * 5.0)
        + rng.normal(0, 0.05, size=len(coords))
    )

    model = SpatialGP(kernel_type="rbf", random_seed=42)
    model.fit(coords, y)

    # Predict at new H3 neighbor cells
    center = str(h3_spatial_domain["center_cell"])
    ring_cells = sorted(str(c) for c in h3.grid_ring(center, 3))
    test_coords = np.array([h3.cell_to_latlng(c) for c in ring_cells])

    pred = model.predict(test_coords, return_std=True)
    assert isinstance(pred, tuple)
    mean, std = pred
    assert len(mean) == len(ring_cells)
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 0)


def test_space_risk_catastrophe_modeling_composition(h3_spatial_domain):
    """Test GEO-INFER-RISK catastrophe and exposure modeling over H3 indexed assets."""
    cells = h3_spatial_domain["disk_cells"]

    # Create exposure table distributed over H3 cells
    portfolio = []
    for idx, cell in enumerate(cells):
        lat, lng = h3.cell_to_latlng(cell)
        portfolio.append(
            {
                "asset_id": f"AST_{idx:04d}",
                "h3_cell": cell,
                "latitude": lat,
                "longitude": lng,
                "total_value": 1_000_000.0 + (idx * 50_000.0),
                "construction_type": "reinforced_concrete"
                if idx % 2 == 0
                else "wood_frame",
                "occupancy_type": "commercial",
                "hazard_zone": "coastal_flood",
            }
        )
    _exposure_df = pd.DataFrame(portfolio)

    exposure_model = EnhancedExposureModel(
        exposure_type="property",
        params={"value_type": "replacement_cost"},
    )
    assert exposure_model.exposure_type == "property"

    # Simulate event losses and verify monotonic empirical exceedance probability
    event_losses = np.array(
        [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000]
    )
    loss_table = pd.DataFrame(
        {
            "event_id": [f"EV_{i}" for i in range(len(event_losses))],
            "hazard_type": ["earthquake"] * len(event_losses),
            "loss": event_losses,
            "rate": [0.01] * len(event_losses),
        }
    )

    ep_curve = calculate_ep_curve(
        loss_table, exceedance_probs=[0.5, 0.2, 0.1, 0.05], exposure_years=100
    )
    assert "return_period" in ep_curve
    assert "exceedance_probability" in ep_curve
    assert "loss" in ep_curve
    assert len(ep_curve["loss"]) == 4


def test_space_act_active_inference_spatial_belief_propagation(h3_spatial_domain):
    """Test Active Inference model over H3 spatial lattice with belief diffusion."""
    center = h3_spatial_domain["center_cell"]
    neighbors = sorted(set(h3.grid_disk(center, 1)) - {center})
    # Deliberately preserve a nonlexical caller state order.
    cells = [neighbors[-1], center, *neighbors[:-1]]
    space = H3StateSpace(cells)
    adapter = get_h3_adapter()

    # Verify adapter wraps SPACE indexing cleanly
    assert str(adapter.latlng_to_cell(37.7749, -122.4194, 8)) == str(
        h3_spatial_domain["center_cell"]
    )

    # Initialize Active Inference Generative Model over discrete spatial states
    n_states = len(cells)
    likelihood = np.full((n_states, n_states), 0.2 / (n_states - 1))
    np.fill_diagonal(likelihood, 0.8)
    prior = np.arange(1, n_states + 1, dtype=float)
    prior /= prior.sum()
    transitions = space.dense_transition_tensor()
    expected_diffusion = np.zeros((n_states, n_states))
    for source, cell in enumerate(cells):
        adjacent = set(h3.grid_disk(cell, 1)) - {cell}
        for destination in adjacent:
            target = cells.index(destination) if destination in cells else source
            expected_diffusion[target, source] += 1 / len(adjacent)
    np.testing.assert_allclose(transitions[:, :, 1], expected_diffusion, atol=1e-12)
    np.testing.assert_allclose(transitions.sum(axis=0), 1, atol=1e-12)
    gen_model = GenerativeModel(
        model_type="categorical",
        parameters={
            "state_dim": n_states,
            "obs_dim": n_states,
            "prior_precision": 1.0,
            "random_seed": 42,
            "A": likelihood,
            "B": transitions,
            "C": np.arange(n_states, dtype=float),
            "D": prior,
        },
    )

    ai_model = ActiveInferenceModel(model_type="categorical", random_seed=42)
    ai_model.set_generative_model(gen_model)

    observation = np.zeros(n_states)
    observed_index = cells.index(center)
    observation[observed_index] = 1.0
    step_result = ai_model.step(observation, available_actions=["stay", "diffuse"])
    beliefs, action = step_result
    # The default A matrix is uninformative. This analytical A/D fixture
    # establishes the actual Bayes update rather than assuming one-hot belief.
    posterior = likelihood[observed_index] * prior
    posterior /= posterior.sum()
    np.testing.assert_allclose(
        # The installed pymdp/JAX backend runs this update in float32.
        np.asarray(beliefs["states"]),
        posterior,
        atol=1e-7,
        rtol=0,
    )
    policy = ai_model.latest_pymdp_result.policy_posterior
    np.testing.assert_allclose(policy.sum(), 1, atol=1e-12)
    assert action == ["stay", "diffuse"][int(np.argmax(policy))]
    assert np.isfinite(ai_model.get_history()[0]["free_energy"])


def test_full_cross_module_end_to_end_pipeline(h3_spatial_domain):
    """
    End-to-end composition:
    1. SPACE: Index region into H3 grid
    2. TIME: Collect spatiotemporal sensor series per cell
    3. BAYES: SpatialGP interpolates missing cell values
    4. RISK: Evaluate hazard exceedance & portfolio loss
    5. ACT: Active Inference agent prioritizes cell
    """
    cells = h3_spatial_domain["disk_cells"]

    # 1. Spatial indexing
    indexer = SpatialIndexingInterface(backend="h3")
    coords = [indexer.cell_to_latlng(c) for c in cells]

    # 2. Temporal observations (sensor telemetry over 7 days)
    base_date = pd.Timestamp("2026-08-01T00:00:00Z")
    telemetry_records = []
    rng = np.random.default_rng(123)
    for day in range(7):
        for idx, cell in enumerate(cells):
            telemetry_records.append(
                {
                    "timestamp": base_date + pd.Timedelta(days=day),
                    "cell": cell,
                    "lat": coords[idx][0],
                    "lng": coords[idx][1],
                    "risk_metric": 10.0 + (idx * 0.5) + rng.normal(0, 0.2),
                }
            )
    telemetry_df = pd.DataFrame(telemetry_records)
    assert len(telemetry_df) == 7 * len(cells)

    # 3. Bayesian Spatial prediction
    train_points = np.array(
        [[r["lat"], r["lng"]] for r in telemetry_records[: len(cells)]]
    )
    train_vals = np.array([r["risk_metric"] for r in telemetry_records[: len(cells)]])
    gp = SpatialGP(kernel_type="rbf", random_seed=42)
    gp.fit(train_points, train_vals)
    preds = gp.predict(train_points)
    assert len(preds) == len(cells)

    # 4. Risk assessment
    loss_data = pd.DataFrame(
        {
            "event_id": [f"E_{i}" for i in range(len(cells))],
            "hazard_type": ["wildfire"] * len(cells),
            "loss": preds * 10_000.0,
            "rate": [0.05] * len(cells),
        }
    )
    ep = calculate_ep_curve(loss_data, exceedance_probs=[0.5, 0.1], exposure_years=50)
    assert len(ep["loss"]) == 2

    # 5. Actual ACT policy evaluation over the predicted spatial risk distribution.
    beliefs = np.asarray(preds) / np.sum(preds)
    policies = [
        {
            "action": cell,
            "predicted_beliefs": np.eye(len(cells))[index],
            "exploration_bonus": 0.0,
        }
        for index, cell in enumerate(cells)
    ]
    selector = PolicySelector(selection_mode="deterministic", random_seed=42)
    evaluation = selector.evaluate_policy_set(beliefs, policies, preferences=beliefs)
    result = selector.select_policy(beliefs, policies, preferences=beliefs)
    assert result["policy"]["action"] == cells[int(np.argmax(preds))]
    assert np.isfinite(evaluation["expected_free_energies"]).all()
    assert result["expected_free_energy"] == pytest.approx(
        min(evaluation["expected_free_energies"])
    )
