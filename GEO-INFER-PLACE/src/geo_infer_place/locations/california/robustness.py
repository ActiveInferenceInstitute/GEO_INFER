"""Tiny synthetic measurement-quality mismatch harness using real monitoring code.

This local research probe is not an accepted experiment or an upstream workflow.
It neither acquires observations nor claims ecological or sensor performance.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

from geo_infer_act.core.gnn_factored_contract import FactoredGNNArtifact
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Hazard,
    ObservatoryConfig,
    Strategy,
    digest,
)
from geo_infer_place.locations.california.geometry import Landscape
from geo_infer_place.locations.california.inference import COSTS
from geo_infer_place.locations.california.models import model_source
from geo_infer_place.locations.california.monitoring import (
    qualify_models,
    run_scenario,
    validate_generator_model,
)

BASELINE_COMMIT = "8bbe89ce409ffedde198be20cc48ae3d5272794e"
QUALITY_FIELDS = (
    "coarse_accuracy",
    "detailed_accuracy",
    "coarse_coverage",
    "detailed_coverage",
)


def research_execution_binding() -> dict[str, object]:
    """Bind the actual PLACE/ACT import path, without claiming a full workflow run."""
    owners = {"geo_infer_place.locations.california.robustness": digest(Path(__file__))}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("geo_infer_place", "geo_infer_act")):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).suffix == ".py":
                owners[name] = digest(Path(filename))
    return {
        "kind": "loaded PLACE/ACT source modules; not full California execution receipt",
        "owners": dict(sorted(owners.items())),
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "pydantic",
                "h3",
                "shapely",
                "pyproj",
                "geopandas",
            )
        },
    }


def generator_conditions(config: ObservatoryConfig) -> dict[str, ObservatoryConfig]:
    """Hold the planner fixed; name mismatch by the planner's relative optimism."""
    return {
        "matched": config,
        "overoptimistic": ObservatoryConfig.model_validate(
            {
                **config.model_dump(),
                "coarse_accuracy": 0.55,
                "detailed_accuracy": 0.65,
                "coarse_coverage": 0.5,
                "detailed_coverage": 0.7,
            }
        ),
        "pessimistic": ObservatoryConfig.model_validate(
            {
                **config.model_dump(),
                "coarse_accuracy": 0.9,
                "detailed_accuracy": 0.995,
                "coarse_coverage": 0.9,
                "detailed_coverage": 0.99,
            }
        ),
    }


def synthetic_generator_models(
    config: ObservatoryConfig, planners: tuple[FactoredGNNArtifact, ...]
) -> tuple[FactoredGNNArtifact, ...]:
    """Construct declared local variants; do not claim independent GNN export."""
    generators = []
    for hazard, planner in zip(Hazard, planners, strict=True):
        source = model_source(config, hazard)
        data = planner.to_dict()
        data.update(json.loads(source))
        data["provenance"] = {
            "producer": "local synthetic robustness variant; not independent GNN export",
            "source_kind": "explicit_factored_json",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        generator = FactoredGNNArtifact.from_dict(data)
        validate_generator_model(planner, generator)
        generators.append(generator)
    return tuple(generators)


def run_tiny_robustness(fixture: Path, models: Path, output: Path) -> Path:
    """Export a create-only, bounded synthetic probe with separate runtime evidence."""
    raw = fixture.read_bytes()
    landscape = Landscape.model_validate_json(raw)
    config = landscape.config
    if (
        len(landscape.cells) != 3
        or config.first_year != 2016
        or config.target_year != 2018
        or config.rounds != 2
        or config.seeds != (0, 1)
        or config.budget != 6
        or config.hazard_budget_floor != 1
        or landscape.source_receipt_sha256 != "0" * 64
        or any(any(item.source_records) for item in landscape.indicators)
        or any(
            getattr(config, field) != getattr(ObservatoryConfig(), field)
            for field in QUALITY_FIELDS
        )
    ):
        raise CaliforniaError("robustness harness requires the tiny synthetic design")
    planners, records = qualify_models(config, models)
    binding = research_execution_binding()
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixture.json").write_bytes(raw)
    retained_models = output / "models"
    retained_models.mkdir()
    for hazard in Hazard:
        for suffix in ("gnn.json", "geo-infer.json"):
            name = f"{hazard.value}.{suffix}"
            (retained_models / f"planner-{name}").write_bytes(
                (models / name).read_bytes()
            )

    traces, timings, assumptions = [], [], {}
    for condition, world_config in generator_conditions(config).items():
        generators = (
            planners
            if condition == "matched"
            else synthetic_generator_models(world_config, planners)
        )
        assumptions[condition] = {
            "planner_config": config.model_dump(mode="json"),
            "generator_config": world_config.model_dump(mode="json"),
            "planner_models": [record.model_dump(mode="json") for record in records],
            "generator_models": [],
        }
        for hazard, generator in zip(Hazard, generators, strict=True):
            source = retained_models / f"{condition}-{hazard.value}.gnn.json"
            artifact = retained_models / f"{condition}-{hazard.value}.geo-infer.json"
            source.write_text(model_source(world_config, hazard), encoding="utf-8")
            artifact.write_text(
                json.dumps(generator.to_dict(), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            assumptions[condition]["generator_models"].append(
                {
                    "hazard": hazard.value,
                    "source_sha256": digest(source),
                    "artifact_sha256": generator.digest,
                    "artifact_file_sha256": digest(artifact),
                    "producer": generator.to_dict()["provenance"]["producer"],
                }
            )
        for seed in config.seeds:
            for strategy in Strategy:
                start = perf_counter()
                scenario = run_scenario(
                    landscape, planners, strategy, seed, generator_artifacts=generators
                )
                elapsed = perf_counter() - start
                identity = {
                    "condition": condition,
                    "seed": seed,
                    "strategy": strategy.value,
                }
                traces.append(
                    {**identity, "scenario": scenario.model_dump(mode="json")}
                )
                timings.append({**identity, "scenario_wall_seconds": elapsed})

    if (
        fixture.read_bytes() != raw
        or qualify_models(config, models)[1] != records
        or research_execution_binding() != binding
    ):
        raise CaliforniaError("input, model or execution bytes changed during probe")
    payload = {
        "schema_version": "local-research/measurement-quality-robustness/1",
        "status": "synthetic smoke probe; no performance-benefit conclusion",
        "baseline_repository": "https://github.com/ActiveInferenceInstitute/GEO_INFER",
        "baseline_commit": BASELINE_COMMIT,
        "license": "CC-BY-NC-SA-4.0; GEO-INFER © 2025 Daniel Ari Friedman (@docxology)",
        "fixture_sha256": digest(output / "fixture.json"),
        "execution": binding,
        "data_kinds": {
            "observed_records": "none; zero source records; no historical acquisition",
            "derived_labels": "thresholds of synthetic values in nominal 2016–2018 slots",
            "simulated_readings": "counterseeded readings from generator likelihoods",
            "inferred_states": "planner posterior probabilities of binary indicator labels",
        },
        "priors": "fixed empirical formula using synthetic 2016–2017 labels only",
        "target": "synthetic 2018 label, used only after allocation and for scoring",
        "pairing": "same seed/hazard/cell/round/mode counter across conditions and strategies",
        "modeled_resource_units": list(COSTS),
        "cost_mismatch_tested": False,
        "runtime": "timings.json; perf_counter around run_scenario only; not a benchmark",
        "site_order": [
            {"hazard": hazard.value, "cell": cell.id}
            for hazard in Hazard
            for cell in landscape.cells
        ],
        "assumptions": assumptions,
        "runs": traces,
    }
    for name, value in (("results.json", payload), ("timings.json", timings)):
        (output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    inventory = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "files.sha256.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    return output / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_tiny_robustness(args.fixture, args.models, args.output))


if __name__ == "__main__":
    main()
