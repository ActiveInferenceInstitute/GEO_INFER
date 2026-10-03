"""Typed contracts for Active Inference scenario runners."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence


SCENARIO_NAMES: tuple[str, ...] = (
    "simple",
    "modern",
    "spatial",
    "h3",
    "ecological",
    "urban_planning",
    "verification",
    "debug",
)

SCENARIO_ALIASES: dict[str, str] = {
    "all": "all",
    "simple_model": "simple",
    "modern_active_inference": "modern",
    "spatial_inference": "spatial",
    "spatial_inference_demo": "spatial",
    "h3_active_inference": "h3",
    "ecological_model": "ecological",
    "urban": "urban_planning",
    "urban-planning": "urban_planning",
    "urban_planning": "urban_planning",
    "verify": "verification",
    "verify_pipeline": "verification",
    "debug_models": "debug",
}


@dataclass
class RunConfig:
    """Configuration for one scenario runner invocation."""

    scenario: str = "simple"
    output_dir: Path | None = None
    seed: int = 42
    deterministic: bool = True
    timesteps: int = 8
    visualizations: bool = True
    h3_resolution: int = 8
    h3_ring_size: int = 1
    h3_cells: list[str] | None = None
    output_formats: list[str] = field(default_factory=lambda: ["json", "csv", "png"])
    parameters: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "geo-infer-act-run-config/v1"

    def __post_init__(self) -> None:
        self.scenario = normalize_scenario_name(self.scenario)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        for name, minimum, maximum in (
            ("seed", 0, None),
            ("timesteps", 1, None),
            ("h3_resolution", 0, 15),
            ("h3_ring_size", 0, None),
        ):
            value = getattr(self, name)
            if (
                type(value) is not int
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(f"{name} must be an integer in its supported range")
        if (
            type(self.deterministic) is not bool
            or type(self.visualizations) is not bool
        ):
            raise ValueError("deterministic and visualizations must be booleans")
        if self.schema_version != "geo-infer-act-run-config/v1":
            raise ValueError("Unsupported run configuration schema_version")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a mapping")
        if self.output_formats != ["json", "csv", "png"]:
            raise ValueError(
                "output_formats must declare the supported json/csv/png bundle; use visualizations=False to omit images"
            )
        self.parameters = copy.deepcopy(self.parameters)
        self.output_formats = list(self.output_formats)
        if self.h3_cells is not None:
            from geo_infer_act.utils.h3_adapter import get_h3_adapter

            adapter = get_h3_adapter()
            self.h3_cells = adapter.validate_cells(self.h3_cells)
            if not self.h3_cells or any(
                adapter.get_resolution(cell) != self.h3_resolution
                for cell in self.h3_cells
            ):
                raise ValueError("h3_cells must be nonempty and match h3_resolution")

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration snapshot."""
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "timesteps": self.timesteps,
            "visualizations": self.visualizations,
            "h3_resolution": self.h3_resolution,
            "h3_ring_size": self.h3_ring_size,
            "h3_cells": copy.deepcopy(self.h3_cells),
            "output_formats": list(self.output_formats),
            "parameters": copy.deepcopy(self.parameters),
        }


@dataclass
class ScenarioRunResult:
    """Result of one scenario runner invocation."""

    scenario: str
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    generated_files: list[Path]


@dataclass
class SuiteRunResult:
    """Result of a multi-scenario runner invocation."""

    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    scenario_results: list[ScenarioRunResult]


def normalize_scenario_name(name: str) -> str:
    """Normalize CLI and script aliases to canonical scenario names."""
    normalized = str(name).strip().replace("-", "_")
    normalized = SCENARIO_ALIASES.get(normalized, normalized)
    if normalized != "all" and normalized not in SCENARIO_NAMES:
        valid = ", ".join(SCENARIO_NAMES)
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid}")
    return normalized


def normalize_scenario_list(names: Iterable[str] | None) -> Sequence[str]:
    """Normalize an optional scenario list."""
    if names is None:
        return SCENARIO_NAMES
    selected = [normalize_scenario_name(name) for name in names]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(
            "Scenario selection must be nonempty and contain no duplicates"
        )
    if "all" in selected:
        return SCENARIO_NAMES
    return selected
