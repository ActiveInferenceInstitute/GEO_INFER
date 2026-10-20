"""Trace-derived monitoring scores, area coverage and descriptive paired contrasts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from geo_infer_place.locations.california.contracts import CaliforniaError, Hazard, Strategy
from geo_infer_place.locations.california.geometry import Landscape
from geo_infer_place.locations.california.inference import COSTS
from geo_infer_place.locations.california.monitoring import Experiment, score_beliefs


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One hazard and round; cumulative coverage counts each selected cell once."""

    seed: int
    strategy: str
    round: int
    hazard: str
    brier: float
    area_weighted_brier: float
    log_score: float
    mean_entropy: float
    spent: int
    cumulative_spent: int
    selected_cells: int
    unique_selected_cells: int
    selected_area_fraction: float
    cumulative_selected_area_fraction: float


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """A within-seed final contrast; positive advantage means lower Brier loss."""

    seed: int
    strategy: str
    hazard: str
    initial_brier: float
    final_brier: float
    brier_reduction: float
    final_area_weighted_brier: float
    rotating_brier_advantage: float
    cumulative_spent: int
    cumulative_selected_area_fraction: float


def evaluate_experiment(
    landscape: Landscape, experiment: Experiment
) -> tuple[tuple[EvaluationRow, ...], tuple[ComparisonRow, ...]]:
    """Recompute recorded scores and summarize simulated monitoring without pooling hazards."""
    count = len(landscape.cells)
    if experiment.config != landscape.config or any(
        len(field.posterior) != count
        for scenario in experiment.scenarios
        for frame in scenario.frames
        for field in frame.fields
    ):
        raise CaliforniaError("evaluation landscape differs from the experiment carrier")
    areas = np.asarray([cell.area_km2 for cell in landscape.cells])
    rows: list[EvaluationRow] = []
    for scenario in experiment.scenarios:
        seen = np.zeros((len(Hazard), count), dtype=bool)
        cumulative = [0] * len(Hazard)
        for frame in scenario.frames:
            for index, (indicator, field) in enumerate(
                zip(landscape.indicators, frame.fields, strict=True)
            ):
                posterior = np.asarray(field.posterior)
                truth = np.asarray(indicator.positive[-1])
                scores = score_beliefs(posterior, truth)
                if not np.allclose(
                    tuple(scores.model_dump().values()),
                    tuple(field.scores.model_dump().values()),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise CaliforniaError("recorded scores differ from posterior and target labels")
                modes = frame.modes[index * count : (index + 1) * count]
                selected = np.asarray(modes) > 0 if modes else np.zeros(count, dtype=bool)
                spent = sum(COSTS[mode] for mode in modes)
                seen[index] |= selected
                cumulative[index] += spent
                rows.append(
                    EvaluationRow(
                        seed=scenario.seed,
                        strategy=scenario.strategy.value,
                        round=frame.round,
                        hazard=field.hazard.value,
                        brier=scores.brier,
                        area_weighted_brier=float(np.average((posterior - truth) ** 2, weights=areas)),
                        log_score=scores.log_score,
                        mean_entropy=scores.mean_entropy,
                        spent=spent,
                        cumulative_spent=cumulative[index],
                        selected_cells=int(selected.sum()),
                        unique_selected_cells=int(seen[index].sum()),
                        selected_area_fraction=float(areas[selected].sum() / areas.sum()),
                        cumulative_selected_area_fraction=float(areas[seen[index]].sum() / areas.sum()),
                    )
                )
    lookup = {(row.seed, row.strategy, row.hazard, row.round): row for row in rows}
    comparisons = []
    for seed in experiment.config.seeds:
        for strategy in Strategy:
            for hazard in Hazard:
                initial = lookup[seed, strategy.value, hazard.value, 0]
                final = lookup[seed, strategy.value, hazard.value, experiment.config.rounds]
                rotating = lookup[seed, Strategy.ROTATING.value, hazard.value, experiment.config.rounds]
                comparisons.append(
                    ComparisonRow(
                        seed=seed,
                        strategy=strategy.value,
                        hazard=hazard.value,
                        initial_brier=initial.brier,
                        final_brier=final.brier,
                        brier_reduction=initial.brier - final.brier,
                        final_area_weighted_brier=final.area_weighted_brier,
                        rotating_brier_advantage=rotating.brier - final.brier,
                        cumulative_spent=final.cumulative_spent,
                        cumulative_selected_area_fraction=final.cumulative_selected_area_fraction,
                    )
                )
    return tuple(rows), tuple(comparisons)


def export_evaluation(
    rows: tuple[EvaluationRow, ...], comparisons: tuple[ComparisonRow, ...], output: Path
) -> None:
    """Write portable CSV tables and a Markdown interpretation of the exact trace."""
    for filename, records, schema in (
        ("trajectory.csv", rows, EvaluationRow),
        ("comparison.csv", comparisons, ComparisonRow),
    ):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(schema)])
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
    lines = [
        "# California monitoring comparison",
        "",
        "Historical source-derived indicator labels; all sensor readings are simulated.",
        "Positive Brier reduction means improvement from the initial spatial belief. ",
        "Positive rotating advantage means lower final Brier loss than rotating allocation ",
        "for the same seed and hazard. These are descriptive paired simulation contrasts, ",
        "not confidence intervals, causal effects, forecasts or field calibration.",
        "",
        "Selected-area coverage records attempted observation, including missing sensor reports. ",
        "Cumulative coverage counts each clipped cell once; resource expenditure counts repeat visits. ",
        "Area-weighted loss uses clipped EPSG:3310 cell areas; cell-average loss weights cells equally.",
        "",
        "| Seed | Strategy | Indicator | Initial Brier | Final Brier | Reduction | Rotating advantage | Units | Selected area |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row.seed} | {row.strategy} | {row.hazard} | {row.initial_brier:.6f} | "
        f"{row.final_brier:.6f} | {row.brier_reduction:.6f} | {row.rotating_brier_advantage:.6f} | "
        f"{row.cumulative_spent} | {row.cumulative_selected_area_fraction:.2%} |"
        for row in comparisons
    )
    with (output / "COMPARISON.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
