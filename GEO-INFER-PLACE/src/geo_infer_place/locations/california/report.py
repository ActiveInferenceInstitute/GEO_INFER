"""Source-bound California maps, graph diagnostics and allocation comparisons."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
import numpy as np
from pyproj import Transformer
from shapely import get_parts
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import transform

from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Hazard,
    Strategy,
    digest,
    strict_bytes,
)
from geo_infer_place.locations.california.geometry import Landscape
from geo_infer_place.locations.california.monitoring import Experiment, check_experiment
from geo_infer_place.locations.california.evaluation import evaluate_experiment, export_evaluation
from geo_infer_place.locations.california.report_integrity import check_report
from geo_infer_space.core.graph_statistics import graph_moran, weighted_group_means

BACKGROUND = "#f7f5ef"
INK = "#162e3b"
COLORS = {
    Strategy.ACTIVE: "#e07643",
    Strategy.QUOTA_ACTIVE: "#168c8d",
    Strategy.ROTATING: "#6a75aa",
}
PROJECT = Transformer.from_crs(4326, 3310, always_xy=True)


def indicator_labels(landscape: Landscape) -> dict[Hazard, str]:
    return {
        Hazard.FIRE: "Mapped fire perimeter",
        Hazard.SEISMIC: f"M{landscape.config.earthquake_magnitude:g}+ catalogued earthquake",
        Hazard.DROUGHT: f"Year-end D{landscape.config.drought_category}+ drought",
    }


def graph_diagnostic(landscape: Landscape, values: np.ndarray) -> dict:
    if values.shape != (len(landscape.cells),) or not np.all(np.isfinite(values)):
        raise CaliforniaError(
            "graph diagnostic values differ from the finite landscape carrier"
        )
    if len(values) < 3:
        return {
            "defined": False,
            "reason": "fewer-than-three-cells",
            "statistics": None,
        }
    if not landscape.edges:
        return {"defined": False, "reason": "no-adjacency-edges", "statistics": None}
    if np.all(values == values[0]):
        return {"defined": False, "reason": "constant-indicator", "statistics": None}
    return {
        "defined": True,
        "reason": None,
        "statistics": asdict(
            graph_moran(values, landscape.edges, permutations=199, seed=0)
        ),
    }


def _patches(landscape: Landscape) -> tuple[list[PathPatch], list[int]]:
    patches, indices = [], []
    for index, cell in enumerate(landscape.cells):
        geometry = transform(PROJECT.transform, shape(cell.geometry.model_dump()))
        for polygon in get_parts(geometry):
            polygon = orient(polygon, sign=1)
            rings = [
                np.asarray(ring.coords)[:, :2]
                for ring in (polygon.exterior, *polygon.interiors)
            ]
            paths = [
                PlotPath(
                    ring,
                    [PlotPath.MOVETO]
                    + [PlotPath.LINETO] * (len(ring) - 2)
                    + [PlotPath.CLOSEPOLY],
                )
                for ring in rings
            ]
            patches.append(PathPatch(PlotPath.make_compound_path(*paths)))
            indices.append(index)
    return patches, indices


def _map(
    axis: Axes,
    landscape: Landscape,
    values: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    vmax: float = 1,
) -> PatchCollection:
    patches, indices = _patches(landscape)
    collection = PatchCollection(
        patches, cmap=cmap, norm=Normalize(0, vmax), edgecolor="#aab5b6", linewidth=0.12
    )
    collection.set_array(values[indices])
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=15, color=INK, pad=14)
    return collection


def map_plate(
    landscape: Landscape,
    values: tuple[np.ndarray, ...],
    titles: tuple[str, ...],
    path: Path,
    *,
    heading: str,
    subtitle: str,
    categorical: bool = False,
    sensor_modes: bool = False,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), facecolor=BACKGROUND)
    for axis, vector, title in zip(axes, values, titles, strict=True):
        axis.set_facecolor(BACKGROUND)
        image = _map(
            axis,
            landscape,
            vector,
            title=title,
            cmap="YlOrBr" if categorical else "viridis",
        )
        if categorical or sensor_modes:
            colors = (
                ["#e3e8e6", "#a95724", "#168c8d"]
                if sensor_modes
                else ["#e3e8e6", "#a95724"]
            )
            image.set_cmap(ListedColormap(colors))
            image.set_norm(BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors)))
        bar = figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            ticks=[0, 1, 2]
            if sensor_modes
            else ([0, 1] if categorical else [0, 0.5, 1]),
        )
        if sensor_modes:
            bar.ax.set_xticklabels(["Passive", "Coarse", "Detailed"])
        bar.set_label(
            "Selected sensor action (resource units: 0 / 1 / 3)"
            if sensor_modes
            else (
                "Historical threshold indicator"
                if categorical
                else "Posterior probability of the threshold indicator"
            ),
            color=INK,
            fontsize=10,
        )
    figure.suptitle(
        heading, x=0.06, y=0.95, ha="left", fontsize=29, color=INK, weight="bold"
    )
    figure.text(0.06, 0.90, subtitle, fontsize=13, color="#546872")
    figure.text(
        0.06,
        0.045,
        f"California county union • H3 resolution {landscape.config.resolution} • California Albers EPSG:3310\nDerived USDM layer: NDMC / USDA / NOAA / NASA. Sources and qualifications in the accompanying ledger.",
        fontsize=10,
        color="#546872",
    )
    figure.subplots_adjust(left=0.05, right=0.95, top=0.83, bottom=0.13, wspace=0.12)
    figure.savefig(path, dpi=100, facecolor=BACKGROUND)
    plt.close(figure)


def group_summaries(landscape: Landscape, values: np.ndarray) -> dict[str, list[dict]]:
    areas = np.asarray([cell.area_km2 for cell in landscape.cells])
    parents = weighted_group_means(
        values, tuple(cell.parent for cell in landscape.cells), areas
    )
    county_values, county_areas, county_ids = [], [], []
    for index, cell in enumerate(landscape.cells):
        for share in cell.county_shares:
            county_values.append(values[index])
            county_areas.append(cell.area_km2 * share.fraction)
            county_ids.append(share.fips)
    counties = weighted_group_means(
        np.asarray(county_values), tuple(county_ids), np.asarray(county_areas)
    )
    for groups in (parents, counties):
        if not np.isclose(
            sum(item.allocated_area for item in groups), areas.sum(), atol=1e-7, rtol=0
        ) or not np.isclose(
            sum(item.allocated_area * item.weighted_mean for item in groups),
            values @ areas,
            atol=1e-7,
            rtol=0,
        ):
            raise CaliforniaError(
                "hierarchical aggregation failed area or weighted-value conservation"
            )
    return {
        "parents": [asdict(item) for item in parents],
        "counties": [asdict(item) for item in counties],
    }


def build_report(landscape_path: Path, experiment_path: Path, output: Path) -> Path:
    landscape = Landscape.model_validate_json(strict_bytes(landscape_path))
    experiment = Experiment.model_validate_json(
        strict_bytes(experiment_path, limit=256 * 1024 * 1024)
    )
    if (
        experiment.landscape_sha256 != digest(landscape_path)
        or experiment.config != landscape.config
    ):
        raise CaliforniaError("experiment is bound to another landscape or design")
    receipt = check_experiment(experiment_path.parent, current_execution=False)
    if experiment_path.name != "experiment.json" or receipt.files["experiment.json"] != digest(experiment_path):
        raise CaliforniaError("report requires the sealed aggregate experiment")
    trajectory, comparisons = evaluate_experiment(landscape, experiment)
    output.mkdir(parents=True, exist_ok=False)
    export_evaluation(trajectory, comparisons, output)
    labels = indicator_labels(landscape)
    map_plate(
        landscape,
        tuple(np.asarray(indicator.positive[-1]) for indicator in landscape.indicators),
        tuple(labels[hazard] for hazard in Hazard),
        output / "historical_indicators.png",
        heading="Three hazards. Three measured indicators.",
        subtitle=f"{landscape.config.target_year} held-out indicator geography; no composite risk score and no forecast claim",
        categorical=True,
    )
    map_plate(
        landscape,
        tuple(np.asarray(indicator.prior) for indicator in landscape.indicators),
        tuple(labels[hazard] for hazard in Hazard),
        output / "empirical_priors.png",
        heading="Learning from strictly earlier years",
        subtitle=f"{landscape.config.first_year}–{landscape.config.target_year - 1} binary frequencies with declared shrinkage; {landscape.config.target_year} excluded from prior estimation",
    )
    county_names = {county.fips: county.name for county in landscape.counties}
    summaries = []
    for scenario in experiment.scenarios:
        for frame in scenario.frames:
            for field in frame.fields:
                aggregation = group_summaries(landscape, np.asarray(field.posterior))
                for county in aggregation["counties"]:
                    county["name"] = county_names[county["identifier"]]
                summaries.append(
                    {
                        "seed": scenario.seed,
                        "strategy": scenario.strategy.value,
                        "round": frame.round,
                        "hazard": field.hazard.value,
                        "scores": field.scores.model_dump(),
                        **aggregation,
                    }
                )
        if scenario.seed == experiment.config.seeds[0]:
            final = scenario.frames[-1]
            count = len(landscape.cells)
            map_plate(
                landscape,
                tuple(
                    np.asarray(final.modes[index * count : (index + 1) * count])
                    for index in range(3)
                ),
                tuple(labels[hazard] for hazard in Hazard),
                output / f"{scenario.strategy.value}_allocation.png",
                heading=f"{scenario.strategy.value.replace('_', ' ').title()} • final monitoring allocation",
                subtitle=f"Round {final.round} • seed {scenario.seed} • {final.spent}/{experiment.config.budget} shared resource units • actions chosen before observing target labels",
                sensor_modes=True,
            )
            for frame in scenario.frames:
                map_plate(
                    landscape,
                    tuple(np.asarray(field.posterior) for field in frame.fields),
                    tuple(labels[hazard] for hazard in Hazard),
                    output / f"{scenario.strategy.value}_{frame.round:02d}.png",
                    heading=f"{scenario.strategy.value.replace('_', ' ').title()} • monitoring round {frame.round}",
                    subtitle=f"Simulated sensors on historical labels • seed {scenario.seed} • {frame.spent}/{experiment.config.budget} units in this round",
                )
    figure, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), facecolor=BACKGROUND)
    for hazard_index, (hazard, axis) in enumerate(zip(Hazard, axes, strict=True)):
        axis.set_facecolor(BACKGROUND)
        for scenario in experiment.scenarios:
            axis.plot(
                [frame.round for frame in scenario.frames],
                [frame.fields[hazard_index].scores.brier for frame in scenario.frames],
                color=COLORS[scenario.strategy],
                linestyle="-" if scenario.seed == experiment.config.seeds[0] else "--",
                label=f"{scenario.strategy.value.replace('_', ' ')} / seed {scenario.seed}",
                linewidth=2.5,
            )
        axis.set_title(labels[hazard], fontsize=14, color=INK)
        axis.set_xlabel("Monitoring round")
        axis.set_ylabel("Brier score against historical labels")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.legend(fontsize=9, frameon=False)
    figure.suptitle(
        "Monitoring objectives have consequences",
        x=0.06,
        y=0.95,
        ha="left",
        fontsize=29,
        color=INK,
        weight="bold",
    )
    figure.text(
        0.06,
        0.88,
        "Pooled information gain, hazard expenditure floors, and rotating observation use the same shared budget.\nIndependent vertical scales. Simulated sensor seeds illustrate variability, not field calibration or forecasting skill.",
        fontsize=13,
        color="#546872",
    )
    figure.subplots_adjust(left=0.06, right=0.96, top=0.78, bottom=0.14, wspace=0.3)
    figure.savefig(output / "score_comparison.png", dpi=100, facecolor=BACKGROUND)
    plt.close(figure)
    diagnostics = {
        indicator.hazard.value: graph_diagnostic(
            landscape, np.asarray(indicator.positive[-1], dtype=float)
        )
        for indicator in landscape.indicators
    }
    facts = {
        "schema_version": "geo-infer-place/california-report/1",
        "config": experiment.config.model_dump(mode="json"),
        "landscape_sha256": digest(landscape_path),
        "source_receipt_sha256": landscape.source_receipt_sha256,
        "experiment_sha256": digest(experiment_path),
        "cells": len(landscape.cells),
        "edges": len(landscape.edges),
        "parents": len({cell.parent for cell in landscape.cells}),
        "counties": len(landscape.counties),
        "administrative_area_km2": landscape.administrative_area_km2,
        "allocated_cell_area_km2": sum(cell.area_km2 for cell in landscape.cells),
        "uncovered_fraction": landscape.uncovered_fraction,
        "county_overlap_km2": landscape.county_overlap_km2,
        "county_weighting": landscape.county_weighting,
        "geometry_repairs": len(landscape.repairs),
        "raw_county_coverage_ratio": {
            "minimum": min(cell.raw_county_coverage_ratio for cell in landscape.cells),
            "maximum": max(cell.raw_county_coverage_ratio for cell in landscape.cells),
        },
        "indicator_roster": [
            {
                "hazard": indicator.hazard.value,
                "unit": indicator.unit,
                "dates": indicator.dates,
                "source_records": indicator.source_records,
                "positive_cells": [sum(row) for row in indicator.positive],
            }
            for indicator in landscape.indicators
        ],
        "graph_diagnostics": diagnostics,
        "aggregations": summaries,
        "scenario_scores": [
            {
                "seed": scenario.seed,
                "strategy": scenario.strategy.value,
                "frames": [
                    {
                        "round": frame.round,
                        "spent": frame.spent,
                        "hazard_spend": {
                            hazard.value: sum(
                                (0, 1, 3)[mode]
                                for mode in frame.modes[
                                    index * len(landscape.cells) : (index + 1)
                                    * len(landscape.cells)
                                ]
                            )
                            for index, hazard in enumerate(Hazard)
                        },
                        "fields": [
                            {
                                "hazard": field.hazard.value,
                                "scores": field.scores.model_dump(),
                                "residual": field.residual,
                                "iterations": field.iterations,
                                "variational_free_energy": field.variational_free_energy,
                            }
                            for field in frame.fields
                        ],
                    }
                    for frame in scenario.frames
                ],
            }
            for scenario in experiment.scenarios
        ],
        "comparisons": [asdict(row) for row in comparisons],
        "qualifications": [
            "Real source-derived historical indicators; sensor observations are simulated.",
            "Exact local one-step GNN policy evaluation; spatial and global policy factorization are mean-field approximations.",
            "Year-end drought snapshots differ from annual fire and earthquake windows.",
            "County attribution normalizes source overlaps; parent summaries allocate clipped child-cell areas.",
            "Moran randomization conditions on this captured graph and values; it is descriptive and does not identify causal effects.",
        ],
    }
    path = output / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(facts, stream, indent=2, allow_nan=False)
    manifest = {
        item.name: digest(item) for item in sorted(output.iterdir()) if item.is_file()
    }
    with (output / "files.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    check_report(output)
    return path
