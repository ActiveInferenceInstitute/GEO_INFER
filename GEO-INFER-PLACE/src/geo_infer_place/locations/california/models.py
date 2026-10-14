"""Explicit GNN factored sources for three distinct monitoring indicators."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from geo_infer_place.locations.california.contracts import Hazard, ObservatoryConfig


def write_model_sources(config: ObservatoryConfig, output: Path) -> tuple[Path, ...]:
    """Declare sensor assumptions; the separate GNN exporter owns artifact production."""
    output.mkdir(parents=True, exist_ok=False)
    return tuple(_write_source(config, hazard, output) for hazard in Hazard)


def _write_source(config: ObservatoryConfig, hazard: Hazard, output: Path) -> Path:
    path = output / f"{hazard.value}.gnn.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(model_source(config, hazard))
    return path


def model_source(config: ObservatoryConfig, hazard: Hazard) -> str:
    if hazard in Hazard:
        sensor = [
            [
                [
                    1.0,
                    float(Decimal(1) - Decimal(str(config.coarse_coverage))),
                    float(Decimal(1) - Decimal(str(config.detailed_coverage))),
                ]
            ]
            * 2,
            [
                [
                    0.0,
                    config.coarse_coverage * config.coarse_accuracy,
                    config.detailed_coverage * config.detailed_accuracy,
                ],
                [
                    0.0,
                    config.coarse_coverage * (1 - config.coarse_accuracy),
                    config.detailed_coverage * (1 - config.detailed_accuracy),
                ],
            ],
            [
                [
                    0.0,
                    config.coarse_coverage * (1 - config.coarse_accuracy),
                    config.detailed_coverage * (1 - config.detailed_accuracy),
                ],
                [
                    0.0,
                    config.coarse_coverage * config.coarse_accuracy,
                    config.detailed_coverage * config.detailed_accuracy,
                ],
            ],
        ]
        source = dict(
            model_name=f"California {hazard.value} indicator monitoring",
            state_factors=[
                dict(
                    id="indicator", states=["below_threshold", "at_or_above_threshold"]
                ),
                dict(id="sensor_mode", states=["passive", "coarse", "detailed"]),
            ],
            control_factors=[
                dict(id="monitor", actions=["passive", "coarse", "detailed"])
            ],
            modalities=[
                dict(
                    id="indicator_report",
                    outcomes=["no_report", "negative", "positive"],
                    dependencies=[0, 1],
                    likelihood=sensor,
                    preferences=[0.0, 0.0, 0.0],
                ),
                dict(
                    id="resource_use",
                    outcomes=["zero_units", "one_unit", "three_units"],
                    dependencies=[1],
                    likelihood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    preferences=[0.0, -config.sensor_cost, -3 * config.sensor_cost],
                ),
            ],
            transitions=[
                dict(
                    dependencies=[0],
                    control_factor=0,
                    probabilities=[[[1, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]],
                ),
                dict(
                    dependencies=[],
                    control_factor=0,
                    probabilities=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ),
            ],
            initial_joint=[0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            policies=[[[0]], [[1]], [[2]]],
            policy_prior=[1 / 3, 1 / 3, 1 / 3],
        )
        return json.dumps(source, indent=2, allow_nan=False) + "\n"
    raise ValueError("Unsupported California indicator")
